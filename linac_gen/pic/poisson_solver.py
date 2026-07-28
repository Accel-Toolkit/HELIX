"""3D FFT Poisson solver with open boundary conditions (Hockney's method).

Solves nabla^2 phi = -rho/eps_0 on a 3D grid using FFT-based convolution
with a free-space Green's function on a doubled (zero-padded) grid.

Two Green's-function variants are supported:

* ``green_kind="igf"`` (default) — Integrated Green Function (Qiang et al.,
  PRSTAB 9, 044204 (2006)).  The cell-averaged 1/r kernel is computed in
  closed form via an analytic antiderivative.  Converges much faster with
  grid resolution than the point-source variant; standard in OPAL,
  IMPACT, Cheetah.

* ``green_kind="point"`` — Point-source 1/(4πε₀ r) sampled at cell
  centres, with a self-potential regulariser at r=0.  Cheaper to build
  and matches the legacy LG behaviour bit-for-bit when needed for
  regression checks.

The solver precomputes the FFT of the Green's function at construction
time. Each call to :meth:`solve` zero-pads the charge density, performs
a convolution via FFT, and returns the three components of E = -grad(phi)
computed with finite differences.

FFTs are dispatched through :mod:`linac_gen.pic.gpu_backend`, which picks
between a CPU path (``scipy.fft`` with ``workers=-1`` or ``numpy.fft`` if
SciPy is unavailable) and a GPU path (``cupyx.scipy.fft`` in FP64).  The
selection is controlled by ``SpaceChargeConfig.use_gpu`` and overridable by
the ``LINAC_GEN_USE_GPU`` env var.  The FFT of the Green's function is
cached on the chosen device, so only ρ and Ex/Ey/Ez cross the host/device
boundary per solve.
"""
import numpy as np

from linac_gen.core.constants import EPSILON_0, PI
from linac_gen.pic.gpu_backend import (
    _CpuBackend,
    _GpuBackend,
    _MpsBackend,
    _maybe_emit_banner,
    _pick_gpu_backend_class,
    select_backend,
)


def _build_backend(use_gpu: str, grid_cells: int):
    """Pick CPU/cupy/MPS backend for the given use_gpu mode and grid size."""
    name = select_backend(use_gpu, grid_cells=grid_cells)
    if name == "gpu":
        backend = _pick_gpu_backend_class(use_gpu)()
    else:
        backend = _CpuBackend()
    _maybe_emit_banner(backend.name)
    return backend


# ---------------------------------------------------------------------------
# Closed-form antiderivative for the integrated Green's function.
#
# Qiang et al. (PRSTAB 9, 044204, 2006) showed that the integral of the free-
# space 1/r kernel over a rectangular cell admits a closed analytic form.
# Define
#
#     F(x, y, z) = - (z²/2) atan2(xy, z·r) - (y²/2) atan2(xz, y·r)
#                  - (x²/2) atan2(yz, x·r)
#                  + y·z log(x + r) + x·z log(y + r) + x·y log(z + r)
#
# with r = sqrt(x² + y² + z²).  Then for a cell centred at displacement
# (i hₓ, j h_y, k h_z) with extents (hₓ, h_y, h_z),
#
#     ∫_cell 1/r dV = 8-point inclusion–exclusion of F at the cell corners.
#
# The integrated Green's function is this divided by the cell volume,
# carrying the 1/(4πε₀) prefactor of the Coulomb kernel.  The antideriv
# is well-defined for all corner positions we evaluate (corners sit at
# ½-integer multiples of the grid spacing, so x, y, z are never simultaneously
# zero); the atan2 form keeps the formula numerically stable when one of the
# coordinates is small.
# ---------------------------------------------------------------------------
def _antideriv_igf(xp, x, y, z):
    """Closed-form antiderivative F where ∂³F/(∂x∂y∂z) = 1/r.

    Uses Cheetah's atan + asinh formulation, which is odd under
    (x, y, z) → (−x, −y, −z) and therefore produces a kernel symmetric
    under cell-displacement reflection — required for the Hockney
    convolution to yield a physically correct (even) Coulomb interaction.
    The atan2 / log form is also a valid antiderivative but picks up
    branch shifts that break this symmetry.

    ``xp`` is the array module (``numpy`` for CPU backend, ``cupy`` for
    GPU); the math is identical, only the dispatch changes.
    """
    r = xp.sqrt(x * x + y * y + z * z)
    # Corners sit at ±½-integer multiples of (dx, dy, dz), so x, y, z are
    # never simultaneously zero and the ``small denominator`` guards below
    # are only there as a defensive epsilon.
    eps = 1e-300
    F = (
        -0.5 * z * z * xp.arctan(x * y / (z * r + eps))
        - 0.5 * y * y * xp.arctan(x * z / (y * r + eps))
        - 0.5 * x * x * xp.arctan(y * z / (x * r + eps))
        + y * z * xp.arcsinh(x / (xp.sqrt(y * y + z * z) + eps))
        + x * z * xp.arcsinh(y / (xp.sqrt(x * x + z * z) + eps))
        + x * y * xp.arcsinh(z / (xp.sqrt(x * x + y * y) + eps))
    )
    return F


class PoissonSolverFFT:
    """3D FFT Poisson solver with open boundary conditions (Hockney's method)."""

    def __init__(self, grid_min: np.ndarray, grid_max: np.ndarray,
                 n_grid: np.ndarray, use_gpu: str = "auto",
                 green_kind: str = "igf"):
        """Precompute Green's function FFT for the given grid.

        Parameters
        ----------
        grid_min : np.ndarray
            (3,) grid lower bounds.
        grid_max : np.ndarray
            (3,) grid upper bounds.
        n_grid : np.ndarray
            (3,) number of grid points per dimension (integer array).
        use_gpu : {"auto", "cpu", "gpu", "cuda", "mps"}
            Backend selection — see ``linac_gen.pic.gpu_backend.select_backend``.
        green_kind : {"igf", "point"}
            Green's function flavour.  ``"igf"`` (default) uses the
            cell-integrated kernel of Qiang 2006; ``"point"`` uses the
            sampled 1/r kernel and the legacy self-potential regulariser.
        """
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.n_grid = np.asarray(n_grid, dtype=np.int64)
        self.dx = (self.grid_max - self.grid_min) / (self.n_grid - 1).astype(
            np.float64
        )
        if green_kind not in ("igf", "point"):
            raise ValueError(
                f"green_kind must be 'igf' or 'point', got {green_kind!r}"
            )
        self.green_kind = green_kind
        grid_cells = int(self.n_grid.prod())
        self._backend = _build_backend(use_gpu, grid_cells)
        # Lazy-allocated reusable charge buffer on the device (filled & FFT'd
        # per solve).  Avoids a per-solve host alloc + H→D copy in adaptive
        # mode (~5–10 ms saved per kick).  Shape never changes.
        self._rho_padded = None
        self._precompute_greens_fft()

    # ------------------------------------------------------------------
    def update_grid(self, grid_min, grid_max) -> None:
        """Re-target the solver to a new grid extent without rebuilding the
        backend, FFT plan cache, or device buffers.  Used by adaptive grid
        mode to avoid re-instantiating the solver every PIC kick.

        ``n_grid``, ``green_kind``, ``use_gpu`` are unchanged; only the grid
        bounds and per-axis spacing are updated, then the Green's function
        FFT is rebuilt in place on the same device.
        """
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.dx = (self.grid_max - self.grid_min) / (self.n_grid - 1).astype(
            np.float64
        )
        self._precompute_greens_fft()

    # ------------------------------------------------------------------
    def _precompute_greens_fft(self) -> None:
        """Compute FFT of the doubled-grid Green's function (on backend device).

        The Green's function is built directly on the backend's array module
        (cupy on GPU, numpy on CPU), so no host→device transfer is needed.
        """
        if self.green_kind == "igf":
            G = self._build_integrated_greens()
        else:
            G = self._build_point_greens()
        self._G_fft = self._backend.rfftn(G)

    # ------------------------------------------------------------------
    def _build_point_greens(self):
        """Sampled 1/(4πε₀ r) Green's function with self-potential regulariser.

        Built on the backend's array module (numpy on CPU, cupy on GPU) so
        the result is already on-device for the FFT.
        """
        xp = self._backend.xp
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        ii = xp.arange(2 * nx, dtype=xp.float64)
        jj = xp.arange(2 * ny, dtype=xp.float64)
        kk = xp.arange(2 * nz, dtype=xp.float64)
        ii = xp.where(ii < nx, ii, ii - 2 * nx)
        jj = xp.where(jj < ny, jj, jj - 2 * ny)
        kk = xp.where(kk < nz, kk, kk - 2 * nz)

        II, JJ, KK = xp.meshgrid(ii, jj, kk, indexing="ij")
        r = xp.sqrt((II * dx) ** 2 + (JJ * dy) ** 2 + (KK * dz) ** 2)

        G = xp.zeros_like(r)
        mask = r > 0
        G[mask] = 1.0 / (4.0 * PI * EPSILON_0 * r[mask])
        # Self-potential regularisation at r=0 (use Python sqrt on host scalars)
        import math
        self_pot = 1.0 / (
            4.0 * PI * EPSILON_0 * 0.5 * math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        )
        G[~mask] = self_pot
        return G

    # ------------------------------------------------------------------
    def _build_integrated_greens(self):
        """Cell-integrated Green's function (Qiang 2006).

        Each cell at displacement (i, j, k)·(dx, dy, dz) is approximated by
        the volume average of 1/(4πε₀r) over a box of size (dx, dy, dz)
        centred on the cell, computed via the 8-point inclusion–exclusion of
        the analytic antiderivative ``_antideriv_igf``.

        Built on the backend's array module (numpy on CPU, cupy on GPU) so
        the result is already on-device for the FFT.
        """
        xp = self._backend.xp
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        ii = xp.arange(2 * nx, dtype=xp.float64)
        jj = xp.arange(2 * ny, dtype=xp.float64)
        kk = xp.arange(2 * nz, dtype=xp.float64)
        ii = xp.where(ii < nx, ii, ii - 2 * nx)
        jj = xp.where(jj < ny, jj, jj - 2 * ny)
        kk = xp.where(kk < nz, kk, kk - 2 * nz)

        # Cell-corner positions — low and high in each axis.
        xL = (ii - 0.5) * dx
        xH = (ii + 0.5) * dx
        yL = (jj - 0.5) * dy
        yH = (jj + 0.5) * dy
        zL = (kk - 0.5) * dz
        zH = (kk + 0.5) * dz

        # 8 corner combinations of the antiderivative — broadcast via meshgrid.
        XL, YL, ZL = xp.meshgrid(xL, yL, zL, indexing="ij")
        XH, YH, ZH = xp.meshgrid(xH, yH, zH, indexing="ij")

        F_HHH = _antideriv_igf(xp, XH, YH, ZH)
        F_LHH = _antideriv_igf(xp, XL, YH, ZH)
        F_HLH = _antideriv_igf(xp, XH, YL, ZH)
        F_HHL = _antideriv_igf(xp, XH, YH, ZL)
        F_LLH = _antideriv_igf(xp, XL, YL, ZH)
        F_LHL = _antideriv_igf(xp, XL, YH, ZL)
        F_HLL = _antideriv_igf(xp, XH, YL, ZL)
        F_LLL = _antideriv_igf(xp, XL, YL, ZL)

        # Inclusion–exclusion over the 8 corners.
        cell_vol = dx * dy * dz
        G = (
            F_HHH - F_LHH - F_HLH - F_HHL
            + F_LLH + F_LHL + F_HLL - F_LLL
        ) / cell_vol
        G *= 1.0 / (4.0 * PI * EPSILON_0)
        return G

    # ------------------------------------------------------------------
    def solve(
        self, rho: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve Poisson equation and return (Ex, Ey, Ez) on the original grid.

        Parameters
        ----------
        rho : np.ndarray
            (nx, ny, nz) charge density array.

        Returns
        -------
        Ex, Ey, Ez : np.ndarray
            (nx, ny, nz) electric field components, always on the host as
            NumPy arrays regardless of backend.
        """
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        # Cell volume (needed to turn density into charge for the convolution)
        cell_vol = dx * dy * dz

        # Zero-pad rho into a reusable on-device buffer.  Allocated lazily on
        # first solve so the shape is known; reused on subsequent solves to
        # avoid the per-kick host alloc + H→D copy.  Shape is fixed by config.
        xp = self._backend.xp
        if self._rho_padded is None:
            self._rho_padded = xp.zeros((2 * nx, 2 * ny, 2 * nz), dtype=xp.float64)
        else:
            self._rho_padded.fill(0.0)
        rho_dev = self._backend.to_device(rho * cell_vol)  # (nx, ny, nz) → device
        self._rho_padded[:nx, :ny, :nz] = rho_dev

        # Convolution via FFT on the chosen backend.  Green's-function FFT
        # is pre-cached on device so only rho / phi cross the H↔D boundary.
        phi_dev = self._backend.irfftn(
            self._backend.rfftn(self._rho_padded) * self._G_fft,
            s=(2 * nx, 2 * ny, 2 * nz),
        )
        phi_padded = self._backend.to_host(phi_dev)

        # Extract potential on original grid
        phi = phi_padded[:nx, :ny, :nz]

        # E = -grad(phi) via finite differences
        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        Ez = np.zeros_like(phi)

        # Central differences (interior)
        Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
        Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
        Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)

        # One-sided differences at boundaries
        Ex[0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / dx
        Ex[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / dx
        Ey[:, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / dy
        Ey[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / dy
        Ez[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dz
        Ez[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dz

        return Ex, Ey, Ez

    # ------------------------------------------------------------------
    def free_memory(self) -> None:
        """Release any device-side memory the backend is holding (no-op on CPU)."""
        self._backend.free_memory()
