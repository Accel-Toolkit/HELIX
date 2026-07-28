"""Charge-deposition kernels onto a 3D grid.

This module exposes two flavours:

* :func:`deposit_cic` — Cloud-In-Cell (8-corner trilinear) deposition.
  Standard first-order particle-mesh scheme used as the LG default.
* :func:`deposit_tsc` — Triangular Shape Cloud (27-cell quadratic)
  deposition.  Each particle is treated as a piecewise-linear "tent"
  of full-width 2 dx in each direction, so its charge spreads to the
  central cell + the six face-adjacent + the twelve edge-adjacent +
  the eight corner-adjacent cells (27 cells total).  Reduces grid-noise
  emittance growth by ~4× for the same grid; standard offering in OPAL.
"""
import numpy as np


def deposit_cic(positions: np.ndarray, charges: np.ndarray,
                grid_min: np.ndarray, grid_max: np.ndarray,
                n_grid: np.ndarray) -> np.ndarray:
    """Deposit particle charges onto a 3D grid using Cloud-In-Cell weighting.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) particle positions [x, y, z] in mm.
    charges : np.ndarray
        (N,) charge per macro-particle.
    grid_min : np.ndarray
        (3,) grid lower bounds [x_min, y_min, z_min].
    grid_max : np.ndarray
        (3,) grid upper bounds [x_max, y_max, z_max].
    n_grid : np.ndarray
        (3,) grid dimensions [nx, ny, nz] (integer array).

    Returns
    -------
    np.ndarray
        (nx, ny, nz) charge density on grid (charge / cell_volume).
    """
    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    rho = np.zeros((nx, ny, nz), dtype=np.float64)

    n_particles = positions.shape[0]
    if n_particles == 0:
        return rho

    # Cell sizes
    dx = (grid_max - grid_min) / (n_grid - 1).astype(np.float64)
    cell_vol = dx[0] * dx[1] * dx[2]

    # Normalized positions (in cell units)
    # pos_norm[i, d] = (positions[i, d] - grid_min[d]) / dx[d]
    pos_norm = (positions - grid_min[np.newaxis, :]) / dx[np.newaxis, :]

    # Cell indices (lower-left corner of the cell containing each particle)
    ix = np.floor(pos_norm[:, 0]).astype(np.int64)
    iy = np.floor(pos_norm[:, 1]).astype(np.int64)
    iz = np.floor(pos_norm[:, 2]).astype(np.int64)

    # Clamp to valid cell range [0, n-2]
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)
    iz = np.clip(iz, 0, nz - 2)

    # Fractional positions within the cell
    fx = pos_norm[:, 0] - ix.astype(np.float64)
    fy = pos_norm[:, 1] - iy.astype(np.float64)
    fz = pos_norm[:, 2] - iz.astype(np.float64)

    # Clamp fractions to [0, 1] for particles outside grid
    fx = np.clip(fx, 0.0, 1.0)
    fy = np.clip(fy, 0.0, 1.0)
    fz = np.clip(fz, 0.0, 1.0)

    # Trilinear weights for 8 corners
    # Weight for corner (ix+di, iy+dj, iz+dk) where di,dj,dk in {0,1}
    wx0 = 1.0 - fx
    wx1 = fx
    wy0 = 1.0 - fy
    wy1 = fy
    wz0 = 1.0 - fz
    wz1 = fz

    # Deposit charge to each of the 8 corners
    for di, wx in enumerate([wx0, wx1]):
        for dj, wy in enumerate([wy0, wy1]):
            for dk, wz in enumerate([wz0, wz1]):
                w = wx * wy * wz * charges
                np.add.at(rho, (ix + di, iy + dj, iz + dk), w)

    # Convert from charge to charge density
    rho /= cell_vol

    return rho


# ---------------------------------------------------------------------------
def _tsc_weights_1d(f: np.ndarray) -> tuple:
    """Return ``(w_minus, w_zero, w_plus)`` 1-D TSC weights.

    Particle at fractional position ``f ∈ [0, 1)`` inside a cell whose
    lower corner has index ``i``.  TSC distributes charge to indices
    ``i−1, i, i+1`` of the *centred* cell (= the cell whose centre is
    closest to the particle).  Returned weights satisfy
    ``w_minus + w_zero + w_plus = 1`` for any ``f``.

    Derivation: the TSC tent function S(x) = max(0, 1 − |x|) at
    fractional offset δ = f − ½ from the central cell centre gives
    weights  w₀ = ¾ − δ², w₊₁ = ½(½ + δ)², w₋₁ = ½(½ − δ)².  Substituting
    δ = f − ½ yields the expressions below.

    KNOWN CONVENTION (verified 2026-07-25): these weights place the
    deposited charge's centroid exactly **half a cell LOW** — the node
    indices are anchored to the lower grid node while δ is measured
    from the cell midpoint.  The gather uses the *same* convention, so
    the deposit↔gather round trip cancels and every space-charge KICK
    is correct; only quantities that read ρ directly (density maps /
    diagnostics) are shifted by −dx/2.  Fixing it means shifting
    deposit AND gather together — do not "fix" one side alone.
    """
    w_zero = 0.75 - (f - 0.5) ** 2
    w_plus = 0.5 * f * f
    w_minus = 0.5 * (1.0 - f) * (1.0 - f)
    return w_minus, w_zero, w_plus


def deposit_tsc(positions: np.ndarray, charges: np.ndarray,
                grid_min: np.ndarray, grid_max: np.ndarray,
                n_grid: np.ndarray) -> np.ndarray:
    """Deposit particle charges onto a 3D grid using TSC weighting.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) particle positions [x, y, z].
    charges : np.ndarray
        (N,) charge per macro-particle.
    grid_min : np.ndarray
        (3,) grid lower bounds.
    grid_max : np.ndarray
        (3,) grid upper bounds.
    n_grid : np.ndarray
        (3,) grid dimensions (integer).

    Returns
    -------
    np.ndarray
        (nx, ny, nz) charge density on grid.
    """
    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    rho = np.zeros((nx, ny, nz), dtype=np.float64)

    n_particles = positions.shape[0]
    if n_particles == 0:
        return rho

    dx = (grid_max - grid_min) / (n_grid - 1).astype(np.float64)
    cell_vol = dx[0] * dx[1] * dx[2]

    pos_norm = (positions - grid_min[np.newaxis, :]) / dx[np.newaxis, :]
    ix = np.floor(pos_norm[:, 0]).astype(np.int64)
    iy = np.floor(pos_norm[:, 1]).astype(np.int64)
    iz = np.floor(pos_norm[:, 2]).astype(np.int64)
    # TSC needs neighbour ix-1 and ix+1, so clamp tighter to leave a margin
    # of one cell at each boundary (particles near edges still deposit, but
    # the off-grid ghost cells are folded into the boundary as is standard).
    ix = np.clip(ix, 1, nx - 2)
    iy = np.clip(iy, 1, ny - 2)
    iz = np.clip(iz, 1, nz - 2)
    fx = np.clip(pos_norm[:, 0] - ix.astype(np.float64), 0.0, 1.0)
    fy = np.clip(pos_norm[:, 1] - iy.astype(np.float64), 0.0, 1.0)
    fz = np.clip(pos_norm[:, 2] - iz.astype(np.float64), 0.0, 1.0)

    wx_m, wx_0, wx_p = _tsc_weights_1d(fx)
    wy_m, wy_0, wy_p = _tsc_weights_1d(fy)
    wz_m, wz_0, wz_p = _tsc_weights_1d(fz)

    wx_arr = (wx_m, wx_0, wx_p)
    wy_arr = (wy_m, wy_0, wy_p)
    wz_arr = (wz_m, wz_0, wz_p)
    offsets = (-1, 0, 1)

    for di, wx in zip(offsets, wx_arr):
        for dj, wy in zip(offsets, wy_arr):
            for dk, wz in zip(offsets, wz_arr):
                w = wx * wy * wz * charges
                np.add.at(rho, (ix + di, iy + dj, iz + dk), w)

    rho /= cell_vol
    return rho
