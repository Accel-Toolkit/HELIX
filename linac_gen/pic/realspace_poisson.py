"""Real-space 3-D Poisson solver with Dirichlet BC (7-point stencil).

Solves  ∇²φ = −ρ/ε₀  on a regular Cartesian grid with φ = 0 on the
exterior box.  Intended use cases:

* Projection-style applications where periodic FFT BCs are wrong
  (e.g. Helmholtz projection of imperfect EM-solver field maps to
  enforce ∇·B = 0 only inside the field-map volume).
* Validation against the FFT Hockney solver in the bulk; should agree
  away from the boundary.

Implementation: scipy.sparse 7-point Laplacian discretisation,
preconditioned conjugate gradient (CG) via :func:`scipy.sparse.linalg.cg`.
A diagonal Jacobi preconditioner is used; no extra dependencies are
required.

This is **not** the main space-charge solver — for SC kicks we keep the
FFT Hockney path with the integrated Green function, which is far faster
on regular grids.  Ship this when you genuinely need Dirichlet boundaries.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import cg

from linac_gen.core.constants import EPSILON_0


def _build_laplacian(nx: int, ny: int, nz: int,
                     dx: float, dy: float, dz: float):
    """Sparse 7-point Laplacian for an (nx, ny, nz) grid (Dirichlet via
    the smaller "interior" sub-problem).

    The returned operator acts on the FLATTENED interior unknowns
    (vector of length (nx-2)*(ny-2)*(nz-2)).  Boundary cells are pinned
    to φ = 0 and folded into the right-hand side.
    """
    ix = nx - 2
    iy = ny - 2
    iz = nz - 2
    n = ix * iy * iz
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)
    invdz2 = 1.0 / (dz * dz)
    diag_main = -2.0 * (invdx2 + invdy2 + invdz2) * np.ones(n)

    # Off-diagonals: ±1 in x  (offset 1), ±1 in y  (offset ix), ±1 in z (offset ix*iy).
    off_x = invdx2 * np.ones(n - 1)
    # Zero out connections that wrap from end of one row to start of next
    # (these are non-physical because they cross the y-boundary).
    for k in range(iy * iz):
        idx = (k + 1) * ix - 1
        if idx < n - 1:
            off_x[idx] = 0.0

    off_y = invdy2 * np.ones(n - ix)
    # Zero out wraps across z-slabs
    for k in range(iz):
        for j in range(ix):
            idx = (k + 1) * (ix * iy) - ix + j
            if idx < n - ix:
                off_y[idx] = 0.0

    off_z = invdz2 * np.ones(n - ix * iy)

    A = diags(
        [off_z, off_y, off_x, diag_main, off_x, off_y, off_z],
        offsets=[-ix * iy, -ix, -1, 0, 1, ix, ix * iy],
        shape=(n, n),
        format="csr",
    )
    return A


class PoissonSolverRealSpace:
    """3-D Dirichlet Poisson solver on a regular grid.

    Parameters
    ----------
    grid_min, grid_max : np.ndarray
        (3,) lower / upper grid bounds.
    n_grid : np.ndarray
        (3,) number of nodes per dimension (≥ 3 in each axis).
    tol, maxiter : float, int
        CG convergence parameters (default 1e-8, 5000).
    """

    def __init__(self, grid_min: np.ndarray, grid_max: np.ndarray,
                 n_grid: np.ndarray, tol: float = 1e-8,
                 maxiter: int = 5000):
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.n_grid = np.asarray(n_grid, dtype=np.int64)
        if np.any(self.n_grid < 3):
            raise ValueError(
                f"All grid dims must be ≥ 3, got {self.n_grid.tolist()}"
            )
        self.dx = (self.grid_max - self.grid_min) / (self.n_grid - 1).astype(
            np.float64
        )
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self._A = _build_laplacian(
            int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2]),
            float(self.dx[0]), float(self.dx[1]), float(self.dx[2]),
        )
        # Diagonal Jacobi preconditioner.
        diag = self._A.diagonal()
        diag = np.where(np.abs(diag) > 0, 1.0 / diag, 0.0)
        self._M = diags(diag, format="csr")

    def solve(self, rho: np.ndarray):
        """Return (Ex, Ey, Ez) on the grid for a given charge density.

        ``rho`` has the SI density convention (C/m³); φ is computed in
        SI volts; E = −∇φ is finite-differenced.  The grid boundary is
        held at φ = 0 (Dirichlet).
        """
        nx, ny, nz = (int(v) for v in self.n_grid)
        ix, iy, iz = nx - 2, ny - 2, nz - 2
        if rho.shape != (nx, ny, nz):
            raise ValueError(
                f"rho shape {rho.shape} does not match grid {(nx, ny, nz)}"
            )
        # Right-hand side: -ρ/ε₀ on interior nodes only.
        rhs_3d = -rho[1:-1, 1:-1, 1:-1] / EPSILON_0
        rhs_flat = rhs_3d.reshape(-1)
        sol_flat, info = cg(self._A, rhs_flat, rtol=self.tol,
                            maxiter=self.maxiter, M=self._M)
        if info != 0:
            # Continue but warn — partial convergence still gives a useful
            # field for Helmholtz-type projection use.
            import warnings
            warnings.warn(
                f"PoissonSolverRealSpace CG did not fully converge "
                f"(info={info}); residual may be above tol={self.tol}.",
                RuntimeWarning, stacklevel=2,
            )
        phi = np.zeros((nx, ny, nz), dtype=np.float64)
        phi[1:-1, 1:-1, 1:-1] = sol_flat.reshape(ix, iy, iz)

        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        Ez = np.zeros_like(phi)
        Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * self.dx[0])
        Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * self.dx[1])
        Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * self.dx[2])
        return Ex, Ey, Ez
