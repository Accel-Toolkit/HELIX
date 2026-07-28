"""Field-interpolation kernels from a 3D grid to particle positions.

* :func:`interpolate_cic` — trilinear (8-corner) gather, the reverse of
  :func:`linac_gen.pic.charge_deposition.deposit_cic`.
* :func:`interpolate_tsc` — quadratic (27-cell) gather using TSC
  weights, the reverse of :func:`deposit_tsc`.

For energy and momentum conservation the interpolation kernel must use
the *same* shape function as the deposition kernel.  Mixing CIC for
deposition with TSC for gather (or vice versa) breaks the discrete
Newton's-third-law symmetry and produces spurious self-forces.
"""
import numpy as np

from linac_gen.pic.charge_deposition import _tsc_weights_1d


def interpolate_cic(
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_z: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    n_grid: np.ndarray,
) -> np.ndarray:
    """CIC field interpolation from grid to particles.

    Parameters
    ----------
    field_x, field_y, field_z : np.ndarray
        (nx, ny, nz) field components on the grid.
    positions : np.ndarray
        (N, 3) particle positions [x, y, z].
    grid_min : np.ndarray
        (3,) grid lower bounds.
    grid_max : np.ndarray
        (3,) grid upper bounds.
    n_grid : np.ndarray
        (3,) grid dimensions (integer array).

    Returns
    -------
    np.ndarray
        (N, 3) interpolated [Fx, Fy, Fz] at each particle position.
    """
    n_particles = positions.shape[0]
    if n_particles == 0:
        return np.zeros((0, 3), dtype=np.float64)

    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])

    # Cell sizes
    dx = (grid_max - grid_min) / (n_grid - 1).astype(np.float64)

    # Normalised positions (in cell units)
    pos_norm = (positions - grid_min[np.newaxis, :]) / dx[np.newaxis, :]

    # Cell indices (lower-left corner)
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

    # Trilinear weights
    wx0 = 1.0 - fx
    wx1 = fx
    wy0 = 1.0 - fy
    wy1 = fy
    wz0 = 1.0 - fz
    wz1 = fz

    # Gather from 8 corners
    result = np.zeros((n_particles, 3), dtype=np.float64)
    for di, wx in enumerate([wx0, wx1]):
        for dj, wy in enumerate([wy0, wy1]):
            for dk, wz in enumerate([wz0, wz1]):
                w = wx * wy * wz  # (N,) weights
                idx_x = ix + di
                idx_y = iy + dj
                idx_z = iz + dk
                result[:, 0] += w * field_x[idx_x, idx_y, idx_z]
                result[:, 1] += w * field_y[idx_x, idx_y, idx_z]
                result[:, 2] += w * field_z[idx_x, idx_y, idx_z]

    return result


# ---------------------------------------------------------------------------
def interpolate_tsc(
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_z: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    n_grid: np.ndarray,
) -> np.ndarray:
    """TSC field interpolation from grid to particles (27-cell stencil).

    Mirror of :func:`deposit_tsc`.  See module docstring for why the
    deposit/gather pair must share the same shape function.
    """
    n_particles = positions.shape[0]
    if n_particles == 0:
        return np.zeros((0, 3), dtype=np.float64)

    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    dx = (grid_max - grid_min) / (n_grid - 1).astype(np.float64)
    pos_norm = (positions - grid_min[np.newaxis, :]) / dx[np.newaxis, :]

    ix = np.floor(pos_norm[:, 0]).astype(np.int64)
    iy = np.floor(pos_norm[:, 1]).astype(np.int64)
    iz = np.floor(pos_norm[:, 2]).astype(np.int64)
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

    result = np.zeros((n_particles, 3), dtype=np.float64)
    for di, wx in zip(offsets, wx_arr):
        for dj, wy in zip(offsets, wy_arr):
            for dk, wz in zip(offsets, wz_arr):
                w = wx * wy * wz
                idx_x = ix + di
                idx_y = iy + dj
                idx_z = iz + dk
                result[:, 0] += w * field_x[idx_x, idx_y, idx_z]
                result[:, 1] += w * field_y[idx_x, idx_y, idx_z]
                result[:, 2] += w * field_z[idx_x, idx_y, idx_z]

    return result
