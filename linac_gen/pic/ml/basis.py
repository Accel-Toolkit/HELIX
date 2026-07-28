"""HALO-PIC delta-rho basis with IGF-precomputed correction fields.

The corrector never predicts a raw field.  It predicts coefficients
c_k of a smooth charge-density basis

    delta_rho(x) = sum_k c_k psi_k(xi),   xi = grid-normalized coords

whose correction field E_corr = sum_k c_k E_k is obtained by solving
each psi_k through the SAME open-boundary IGF Poisson solver the coarse
PIC uses.  Consequences (the referee-driven design):

* Gauss-law consistency: the correction field is the exact field OF a
  charge distribution — no phantom-force freedom.
* Zero net self-force and zero added charge: each psi_k has its
  monopole and dipole moments projected out on the grid, so the
  centroid feels no net kick and total charge is untouched.
* Band-limited by construction: the basis is smooth Hermite–Gaussian
  content in beam-sigma units (the grid is built as mean +/- extent*std,
  so grid-normalized coordinates ARE sigma-scaled).  The net physically
  cannot chase deposition shot noise.
* Cheap: E_k are cached per quantized grid geometry; runtime correction
  is a weighted sum of cached grids.

Coefficients are dimensionless multipliers of basis functions carrying
unit "charge scale" rho0 = Q_bunch / cell_volume_total; the E_k are
stored in the coarse solver's native field units so they add directly
onto the solver output.
"""
from __future__ import annotations

import numpy as np

# Probabilists' Hermite polynomials He_n via recurrence
def _hermite_e(n: int, x: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    hm2, hm1 = np.ones_like(x), x
    for k in range(2, n + 1):
        hm2, hm1 = hm1, x * hm1 - (k - 1) * hm2
    return hm1


def default_mode_indices(max_degree: int = 4) -> list[tuple[int, int, int]]:
    """All (i, j, l) with 2 <= i+j+l <= max_degree.

    Degree-0 (monopole) and degree-1 (dipoles) are excluded up front;
    residual monopole/dipole content from grid truncation is projected
    out numerically in :class:`BasisFieldCache`.
    """
    out = []
    for d in range(2, max_degree + 1):
        for i in range(d + 1):
            for j in range(d - i + 1):
                out.append((i, j, d - i - j))
    return out


class BasisFieldCache:
    """Per-grid-geometry cache of basis charge densities and their
    IGF-solved correction fields.

    One entry per quantized (dx, dy, dz); within an entry:
      rho_k   : (n_basis, nx, ny, nz) basis densities (moment-projected)
      E_k     : (n_basis, 3, nx, ny, nz) fields in solver units
      gram_inv: (n_basis, n_basis) inverse Gram of the E_k (for anchor
                least-squares projection of a measured defect field)
    """

    # Entries are large (E is (n_basis, 3, nx, ny, nz)); a breathing beam
    # crosses several sigma bins, so bound the cache FIFO-style.  16
    # entries at 48^3 is ~1.3 GB worst case, ~50 MB at 24^3.
    max_entries: int = 16

    def __init__(self, mode_indices=None, gauss_width_sigma: float = 1.0,
                 quant: float = 0.10, sigma_quant: float = 0.15):
        self.modes = list(mode_indices or default_mode_indices(4))
        self.width = float(gauss_width_sigma)   # basis Gaussian width in beam sigma
        self.quant = float(quant)               # log-quantization bin for (dx,dy,dz)
        self.sigma_quant = float(sigma_quant)   # log bin for current beam sigma
        self._entries: dict[tuple, dict] = {}

    # -- geometry key ---------------------------------------------------
    def _key(self, grid_min, grid_max, n_grid, sigma) -> tuple:
        d = (np.asarray(grid_max) - np.asarray(grid_min)) / np.maximum(
            np.asarray(n_grid) - 1, 1)
        q = np.round(np.log(np.maximum(d, 1e-12)) / self.quant).astype(int)
        # quantized CURRENT beam sigma: the basis must track the beam as
        # it breathes (a mismatched beam spans ~3x in size; a basis frozen
        # to the kick-0 box has no support where the bloated beam lives)
        qs = np.round(np.log(np.maximum(np.asarray(sigma, float), 1e-12))
                      / self.sigma_quant).astype(int)
        return (int(n_grid[0]), int(n_grid[1]), int(n_grid[2]),
                int(q[0]), int(q[1]), int(q[2]),
                int(qs[0]), int(qs[1]), int(qs[2]))

    def _snap_sigma(self, sigma) -> np.ndarray:
        """Sigma rounded to its quantization bin (what the entry is
        actually built with, so equal keys => identical basis)."""
        qs = np.round(np.log(np.maximum(np.asarray(sigma, float), 1e-12))
                      / self.sigma_quant)
        return np.exp(qs * self.sigma_quant)

    def n_basis(self) -> int:
        return len(self.modes)

    # -- entry construction ----------------------------------------------
    def _build_rho(self, n_grid, grid_min, grid_max, sigma) -> np.ndarray:
        """Basis densities in CURRENT-beam-sigma coordinates.

        u_i = (x_i - box_center_i) / sigma_i, so the basis breathes with
        the beam; the basis is
        He_i(u_x) He_j(u_y) He_l(u_z) exp(-|u|^2 / (2 w^2)).
        """
        gmin = np.asarray(grid_min, float)
        gmax = np.asarray(grid_max, float)
        ctr = 0.5 * (gmin + gmax)
        sig = np.asarray(sigma, float)
        axes = [(np.linspace(gmin[i], gmax[i], int(n_grid[i])) - ctr[i])
                / sig[i] for i in range(3)]
        ux, uy, uz = np.meshgrid(*axes, indexing="ij")
        g = np.exp(-(ux**2 + uy**2 + uz**2) / (2.0 * self.width**2))
        rho = np.empty((len(self.modes),) + tuple(int(n) for n in n_grid))
        for k, (i, j, l) in enumerate(self.modes):
            rho[k] = (_hermite_e(i, ux) * _hermite_e(j, uy)
                      * _hermite_e(l, uz) * g)
        # Project out monopole + dipole content (grid-exact conservation)
        ones = np.ones_like(ux)
        raw_moments = [ones, ux, uy, uz]
        # Gram-Schmidt the moment functions on the grid
        moments = []
        for m in raw_moments:
            v = m.copy()
            for b in moments:
                v -= (v * b).sum() * b
            v /= np.sqrt((v * v).sum())
            moments.append(v)
        for k in range(rho.shape[0]):
            for b in moments:
                rho[k] -= (rho[k] * b).sum() * b
            # normalize each basis density to unit L2 (coefficients then
            # measure defect amplitude on a common scale)
            nrm = np.sqrt((rho[k] ** 2).sum())
            if nrm > 0:
                rho[k] /= nrm
        return rho

    def get(self, solver, grid_min, grid_max, n_grid, sigma) -> dict:
        """Fields for the current (geometry, beam-sigma) pair, building
        on first use.

        ``solver`` is the coarse ``PoissonSolverFFT`` already retargeted
        to (grid_min, grid_max) — reusing it guarantees the basis fields
        live in exactly the coarse solver's units and discretization.
        ``sigma`` is the CURRENT per-axis beam sigma (physical units,
        same as the grid); it is quantized in log bins, so nearby beam
        sizes share one cached entry.
        """
        key = self._key(grid_min, grid_max, n_grid, sigma)
        d_now = ((np.asarray(grid_max, float) - np.asarray(grid_min, float))
                 / np.maximum(np.asarray(n_grid) - 1, 1))
        entry = self._entries.get(key)
        if entry is not None:
            # Log-quantized keys collide for cell sizes up to ~quant apart
            # (adaptive-grid mode resizes the box every kick).  Reusing
            # fields solved at a >2% different cell size mis-scales the
            # correction — rebuild the entry in place instead.
            if np.max(np.abs(np.log(d_now / entry["d"]))) < 0.02:
                return entry
            del self._entries[key]
        rho = self._build_rho(n_grid, grid_min, grid_max,
                              self._snap_sigma(sigma))
        nb = rho.shape[0]
        shape = tuple(int(n) for n in n_grid)
        E = np.empty((nb, 3) + shape)
        for k in range(nb):
            ex, ey, ez = solver.solve(rho[k])
            E[k, 0], E[k, 1], E[k, 2] = ex, ey, ez
        flat = E.reshape(nb, -1)
        gram = flat @ flat.T
        # Tikhonov floor keeps near-degenerate high-order modes stable
        tr = float(np.trace(gram))
        if not np.isfinite(tr) or tr <= 0.0:
            # Degenerate basis (e.g. sigma collapsed far below the grid
            # floor -> all densities underflow).  Disable the correction
            # for this geometry instead of inverting a singular Gram.
            gram_inv = np.zeros_like(gram)
        else:
            gram += np.eye(nb) * (1e-10 * tr / nb)
            gram_inv = np.linalg.inv(gram)
        entry = {"rho": rho, "E": E, "flat": flat, "gram": gram,
                 "gram_inv": gram_inv, "key": key, "d": d_now}
        if len(self._entries) >= self.max_entries:      # FIFO bound
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = entry
        return entry

    # -- runtime ops -----------------------------------------------------
    @staticmethod
    def correction_field(entry: dict, coeffs: np.ndarray) -> tuple:
        """E_corr = sum_k c_k E_k as three (nx,ny,nz) arrays."""
        E = np.tensordot(coeffs, entry["E"], axes=(0, 0))   # (3, nx,ny,nz)
        return E[0], E[1], E[2]

    @staticmethod
    def project_defect(entry: dict, dEx, dEy, dEz) -> np.ndarray:
        """Least-squares coefficients of a measured defect field."""
        d = np.concatenate([np.ravel(dEx), np.ravel(dEy), np.ravel(dEz)])
        b = entry["flat"] @ d
        return entry["gram_inv"] @ b

    @staticmethod
    def cell_weights(rho, floor: float = 0.01) -> np.ndarray:
        """Beam-density cell weights in [floor, 1 + floor].

        ``floor`` keeps a small uniform component so the vacuum region is
        fitted loosely rather than ignored entirely.
        """
        w = np.ravel(np.asarray(rho, float))
        w = np.maximum(w, 0.0)
        wmax = w.max()
        if wmax <= 0:
            return np.ones_like(w)
        return w / wmax + floor

    @staticmethod
    def project_defect_weighted(entry: dict, dEx, dEy, dEz,
                                w: np.ndarray) -> np.ndarray:
        """Beam-density-weighted least squares: fit the defect where the
        charge actually is.  An unweighted fit over the whole box
        down-weights the core — but the core field sets the depressed
        tune, the observable with the tightest gate (0.1 deg/cell).

        ``w`` is per-cell weights from :meth:`cell_weights`.
        """
        ws = np.sqrt(np.tile(w, 3))
        A = entry["flat"] * ws
        d = np.concatenate([np.ravel(dEx), np.ravel(dEy),
                            np.ravel(dEz)]) * ws
        G = A @ A.T
        G += np.eye(G.shape[0]) * (1e-10 * np.trace(G) / G.shape[0])
        return np.linalg.solve(G, A @ d)

    @staticmethod
    def weighted_norm(w: np.ndarray, fx, fy, fz) -> float:
        """sqrt(sum_cells w * |F|^2 / sum w) — the beam-weighted rms
        field magnitude used for defect/residual metrics."""
        s2 = (w * (np.ravel(fx) ** 2 + np.ravel(fy) ** 2
                   + np.ravel(fz) ** 2)).sum()
        return float(np.sqrt(s2 / w.sum()))
