"""Space-charge field reused between Poisson solves.

On a solve kick the field is computed exactly as :class:`PicSolver` computes
it, and its grid values are also kept in beam-size units: the grid in the
normalised coordinates u = (x - <x>) / s and the dimensionless node fields
g_i = E_i s_bar s_i / Q, where s are the rms sizes of the rest-frame
distribution, s_bar their geometric mean and Q the charge of the alive
macro-particles.  On the kicks in between, the kept field is rescaled to
the beam as it is then -- its centre, rms sizes and charge:
E_i = g_i(u) Q / (s_bar s_i), one gather with the same charge-assignment
kernel and no deposit or Poisson solve (a quasi-frozen field in the spirit
of Seletskiy and Banerjee, Space Charge 2026, here with the 3-D PIC field
kept in beam-size units).

The rescaling is exact when the beam only shifts, scales uniformly or
changes its charge between solves.  A change of the aspect ratio is
approximated (the field of a beam stretched along one axis does not follow
1/(s_bar s_i) exactly: a 20 % longer bunch gives a transverse field about
9 % too strong), and a change of the shape in normalised coordinates is
neglected, which is why the field is solved again every ``reuse_every``
kicks and after the events listed in :class:`FieldReusePicSolver`.
"""
from __future__ import annotations

import numpy as np

from linac_gen.pic.pic_solver import PicSolver, _select_kernel


class FieldReusePicSolver(PicSolver):
    """3-D PIC that solves the field on every ``config.reuse_every``-th kick
    and rescales the last solved field in between.

    The kept field is dropped, and the next kick solves, after a
    ``HELIX_SC_GRID`` card is applied, after a kick that took the
    bunch-train path (whose kicks always solve), after more than
    :attr:`max_reuse_travel_mm` of travel or after more than
    :attr:`max_reuse_loss` of the alive macro-particles were lost since the
    last solve, and when the beam has no extent or charge.

    ``reuse_every=1`` is the plain PIC: every kick runs
    :meth:`PicSolver._rest_frame_field` unchanged.
    """

    #: Travel since the last solve after which the field is solved again [mm].
    max_reuse_travel_mm: float = 100.0
    #: Fraction of the alive macro-particles lost since the last solve after
    #: which the field is solved again.
    max_reuse_loss: float = 0.01

    def __init__(self, config):
        super().__init__(config)
        self.reuse_every = int(getattr(config, "reuse_every", 1) or 1)
        # A backward walk visits the kicks in reverse order, so it cannot
        # reproduce which kicks solved and which reused a stored field.
        self.exact_backward = self.reuse_every == 1
        self._store = None
        self._since_solve = 0
        self.n_solves = 0
        self.n_reused = 0
        if self.reuse_every > 1:
            from linac_gen.pic import gpu_particles
            if gpu_particles.requested(config):
                gpu_particles.warn_once(
                    "reuse", "gpu_particles is not used with reuse_every="
                    "%d: field reuse keeps the CPU deposit and gather",
                    self.reuse_every)

    # ------------------------------------------------------------------
    def set_grid_extent(self, extent_sigma: float) -> None:
        super().set_grid_extent(extent_sigma)
        self._store = None

    def _kick_bunch_train(self, beam, ds: float) -> None:
        self._store = None
        super()._kick_bunch_train(beam, ds)

    def stats(self) -> dict:
        """PicSolver's counters plus solved and reused kicks of the bunch
        grid (bunch-train kicks always solve, on the companion grid, and
        count there) and the re-solve limits in force."""
        return {**super().stats(), "solves": self.n_solves,
                "reused": self.n_reused, "reuse_every": self.reuse_every,
                "max_reuse_travel_mm": self.max_reuse_travel_mm,
                "max_reuse_loss": self.max_reuse_loss}

    # ------------------------------------------------------------------
    def _rest_frame_field(self, coords_rest: np.ndarray, beam) -> np.ndarray:
        if self.reuse_every <= 1:
            return super()._rest_frame_field(coords_rest, beam)
        if self._store is not None and self._since_solve < self.reuse_every - 1:
            E = self._reused_field(coords_rest, beam)
            if E is not None:
                self._since_solve += 1
                self.n_reused += 1
                return E
        return self._solve_and_keep(coords_rest, beam)

    # ------------------------------------------------------------------
    def _solve_and_keep(self, coords_rest: np.ndarray, beam) -> np.ndarray:
        """PicSolver's field step (same calls, same order), keeping the grid
        field in beam-size units."""
        self._setup_grid(coords_rest)
        charges = self._macro_charges(beam)
        deposit_fn, interpolate_fn = _select_kernel(
            getattr(self.config, "kernel", "cic")
        )
        rho = deposit_fn(
            coords_rest, charges,
            self._grid_min, self._grid_max, self._n_grid,
        )
        Ex, Ey, Ez = self._solver.solve(rho)
        E = interpolate_fn(
            Ex, Ey, Ez,
            coords_rest, self._grid_min, self._grid_max, self._n_grid,
        )
        self.n_solves += 1
        self._since_solve = 0
        self._store = None
        mu = np.mean(coords_rest, axis=0)
        s = np.std(coords_rest, axis=0)
        Q = float(np.sum(charges))
        if self._has_extent(s) and Q > 0.0:
            s_bar = float(np.prod(s) ** (1.0 / 3.0))
            scale = s_bar * s / Q
            self._store = {
                "g": (Ex * scale[0], Ey * scale[1], Ez * scale[2]),
                "umin": (self._grid_min - mu) / s,
                "umax": (self._grid_max - mu) / s,
                "n_grid": self._n_grid.copy(),
                "gather": interpolate_fn,
                "s_mm": float(beam.ref.s),
                "n_alive": int(beam.n_alive),
            }
        return E

    def _has_extent(self, s: np.ndarray) -> bool:
        """Every axis wide enough for the beam-size units: the PIC grid
        floors its half-width at 1e-6 mm, below which the size is rounding
        noise and the rescaling would divide by it."""
        extent = (self._extent_override if self._extent_override is not None
                  else self.config.grid_extent)
        return bool(np.all(float(extent) * s > 1e-6))

    def _reused_field(self, coords_rest: np.ndarray, beam):
        """The kept field rescaled to the beam now, or None when it must be
        solved again."""
        st = self._store
        if abs(float(beam.ref.s) - st["s_mm"]) > self.max_reuse_travel_mm:
            return None
        if beam.n_alive < (1.0 - self.max_reuse_loss) * st["n_alive"]:
            return None
        mu = np.mean(coords_rest, axis=0)
        s = np.std(coords_rest, axis=0)
        Q = float(np.sum(self._macro_charges(beam)))
        if not (self._has_extent(s) and Q > 0.0):
            return None
        u = (coords_rest - mu) / s
        gx, gy, gz = st["g"]
        g = st["gather"](gx, gy, gz, u, st["umin"], st["umax"], st["n_grid"])
        s_bar = float(np.prod(s) ** (1.0 / 3.0))
        return g * (Q / (s_bar * s))
