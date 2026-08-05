"""Probabilistic Hofmann stability chart under an engineering jitter budget.

Monte-Carlo layer over :mod:`linac_gen.analysis.hofmann_dispersion`: given a
nominal chart point ``(R, Y, eps_z/eps_x)`` and Gaussian jitter budgets for
current, mismatch, emittance ratio, and tunes, computes the probability that
the coherent growth rate exceeds a threshold.  Ported verbatim from the
source manuscript package (``PRAB_Hofmann/analysis/probabilistic.py``);
keep the numerics diffable against that source.

Public API
----------

::

    from linac_gen.analysis.hofmann_probabilistic import (
        ProbabilisticChart, instability_probability,
    )

    pc = ProbabilisticChart()
    R, Y, P = pc.compute_probability_grid(eps_ratio=1.475, steps=50, N_mc=200)
    p_cell = instability_probability(coords)     # per-cell, from the adapter

:func:`instability_probability` is the HELIX-facing per-cell summary used by
the GUI and MCP surfaces (~0.2 s/cell at ``N_mc=200``).  The probability
*grid* and :meth:`ProbabilisticChart.bootstrap_contour` (B x full-grid
resamples — hours-class serial) are headless-only tools; ``n_workers`` farms
the bootstrap over processes with BLAS threads pinned.

Sigma defaults match the source manuscript (current jitter 3%, mismatch 10%,
emittance-ratio scatter 5%, tune jitter 2%).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from linac_gen.analysis.hofmann_dispersion import ChartAborted, DispersionSolver

Y_MIN_DEFAULT = 0.2  # matches the source package's chart floor


@dataclass
class PerturbationBudget:
    """Engineering sensitivity budget for Monte Carlo perturbations."""

    sigma_I: float = 0.03
    sigma_mismatch: float = 0.10
    sigma_eps: float = 0.05
    sigma_tune: float = 0.02


def sample_perturbations(
    R: float,
    Y: float,
    eps_ratio: float,
    N_mc: int,
    budget: PerturbationBudget = PerturbationBudget(),
    rng: Optional[np.random.Generator] = None,
    y_min: float = Y_MIN_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``N_mc`` perturbed operating points at nominal ``(R, Y, eps)``.

    Higher current means stronger space charge means LOWER tune-depression Y
    (sign convention from the legacy "Fix 4").
    """
    if rng is None:
        rng = np.random.default_rng()

    dI = rng.normal(0, budget.sigma_I, N_mc)
    Y_p = Y * (1.0 - dI)

    dM = rng.normal(0, budget.sigma_mismatch, N_mc)
    Y_p *= 1.0 + 0.5 * dM

    dE = rng.normal(0, budget.sigma_eps, N_mc)
    eps_p = np.maximum(eps_ratio * (1.0 + dE), 0.01)

    dR = rng.normal(0, budget.sigma_tune * R, N_mc) if R > 0 else np.zeros(N_mc)
    R_p = np.maximum(R + dR, 1e-6)

    dY = rng.normal(0, budget.sigma_tune * Y, N_mc)
    Y_p = np.clip(Y_p + dY, y_min, 2.0)

    return R_p, Y_p, eps_p


class ProbabilisticChart:
    """Monte Carlo probabilistic stability chart with bootstrap CIs."""

    def __init__(
        self,
        solver: Optional[DispersionSolver] = None,
        budget: PerturbationBudget = PerturbationBudget(),
    ):
        self.solver = solver or DispersionSolver()
        self.budget = budget

    # ---- core grid -----------------------------------------------------------

    def compute_probability_grid(
        self,
        eps_ratio: float,
        r_max: float = 2.0,
        y_max: float = 2.0,
        steps: int = 80,
        N_mc: int = 500,
        threshold: float = 0.01,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = 42,
        y_min: float = Y_MIN_DEFAULT,
        verbose: bool = False,
        should_stop: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(R_arr, Y_arr, P_grid)``.

        ``P_grid[i, j] = P(growth > threshold | R_arr[j], Y_arr[i])`` over
        ``N_mc`` perturbations drawn from the engineering budget.

        Reproducibility: if ``rng`` is ``None`` we construct a fresh
        ``default_rng(seed)`` — this matches the source package.

        ``should_stop`` (checked once per grid row) raises
        :class:`~linac_gen.analysis.hofmann_dispersion.ChartAborted` when it
        returns True; ``progress`` receives ``(rows_done, ny)`` after each
        row.  Both default to None, leaving the computation identical to
        the source package.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        nx = max(40, steps)
        ny = max(28, int(0.55 * steps * (y_max - y_min) / max(r_max, 1e-6)))
        R_arr = np.linspace(1e-6, r_max, nx)
        Y_arr = np.linspace(y_min, y_max, ny)
        P_grid = np.zeros((ny, nx))

        total = nx * ny
        done = 0
        for iy in range(ny):
            if should_stop is not None and should_stop():
                raise ChartAborted("probability-grid computation aborted")
            for ix in range(nx):
                R0, Y0 = R_arr[ix], Y_arr[iy]
                R_p, Y_p, eps_p = sample_perturbations(
                    R0, Y0, eps_ratio, N_mc=N_mc, budget=self.budget,
                    rng=rng, y_min=y_min,
                )
                # Batched solver call (numpy-vectorised over the N_mc
                # perturbations) — replaces the previous per-sample loop
                # and gives a ~80x speedup of compute_probability_grid.
                g = self.solver.growth_rate_batch(R_p, Y_p, eps_p)
                P_grid[iy, ix] = float((g > threshold).sum()) / N_mc
                done += 1
            if progress is not None:
                progress(iy + 1, ny)
            if verbose and ((iy + 1) % 5 == 0 or iy == ny - 1):
                print(f"    row {iy+1}/{ny}  ({100 * done / total:.0f}%)")

        return R_arr, Y_arr, P_grid

    # ---- bootstrap contour CI -----------------------------------------------

    def bootstrap_contour(
        self,
        eps_ratio: float,
        level: float = 0.5,
        r_max: float = 2.0,
        y_max: float = 2.0,
        steps: int = 50,
        N_mc: int = 200,
        B: int = 200,
        threshold: float = 0.01,
        seed: int = 0,
        y_min: float = Y_MIN_DEFAULT,
        verbose: bool = False,
        n_workers: Optional[int] = None,
    ) -> dict:
        """Bootstrap uncertainty band on the ``P=level`` contour.

        For each of ``B`` resamples we draw ``N_mc`` perturbations with a
        fresh seed and recompute ``P_grid``.  We return the pointwise
        16/50/84 percentile maps of ``P`` across the ``B`` resamples.
        The positional uncertainty of the ``P = level`` contour itself
        (its median and worst-case width in ``eta``) is obtained from
        these maps by :func:`contour_band_width`, which interpolates the
        column-wise ``P = level`` crossings of ``P_lower`` and ``P_upper``.

        Parallelism
        -----------
        ``n_workers=None`` (default) uses serial computation.
        ``n_workers=k`` farms the ``B`` bootstrap resamples across ``k``
        processes via ``ProcessPoolExecutor`` — embarrassingly parallel,
        identical numerics, ~k× wall-clock speedup until you saturate
        the physical-core count.  BLAS thread counts in each worker are
        pinned to 1 to avoid oversubscription on a multi-core CPU.

        Returns
        -------
        dict with keys ``R_arr``, ``Y_arr``, ``P_median``, ``P_lower``,
        ``P_upper``, ``level``, ``B``, ``N_mc``.
        """
        master_rng = np.random.default_rng(seed)
        seeds = master_rng.integers(0, 2**31 - 1, size=B)

        if n_workers is not None and n_workers > 1:
            # Pin BLAS thread counts in workers; capture/restore parent env.
            _thread_env_keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                "NUMEXPR_NUM_THREADS")
            prev_env = {k: os.environ.get(k) for k in _thread_env_keys}
            for k in _thread_env_keys:
                os.environ[k] = "1"
            try:
                args_list = [
                    (eps_ratio, r_max, y_max, steps, N_mc, threshold,
                     int(s), y_min, self.budget)
                    for s in seeds
                ]
                with ProcessPoolExecutor(max_workers=n_workers) as ex:
                    stack = list(ex.map(_compute_one_bootstrap_chart,
                                        args_list, chunksize=1))
            finally:
                for k, v in prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            # axes are deterministic from params — recompute locally
            nx = max(40, int(steps))
            ny = max(28, int(0.55 * steps * (y_max - y_min) / max(r_max, 1e-6)))
            R_arr = np.linspace(1e-6, r_max, nx)
            Y_arr = np.linspace(y_min, y_max, ny)
        else:
            stack = []
            R_arr = Y_arr = None
            for b, s in enumerate(seeds):
                R_arr, Y_arr, P = self.compute_probability_grid(
                    eps_ratio=eps_ratio, r_max=r_max, y_max=y_max,
                    steps=steps, N_mc=N_mc, threshold=threshold,
                    seed=int(s), y_min=y_min, verbose=False,
                )
                stack.append(P)
                if verbose and (b + 1) % max(1, B // 10) == 0:
                    print(f"    bootstrap {b+1}/{B}")

        stack_arr = np.stack(stack, axis=0)  # (B, ny, nx)
        return {
            "R_arr": R_arr,
            "Y_arr": Y_arr,
            "P_median": np.quantile(stack_arr, 0.50, axis=0),
            "P_lower":  np.quantile(stack_arr, 0.16, axis=0),
            "P_upper":  np.quantile(stack_arr, 0.84, axis=0),
            "level": level,
            "B": B,
            "N_mc": N_mc,
        }


def _column_crossing(P_col: np.ndarray, Y_arr: np.ndarray, level: float) -> float:
    """First ``P = level`` crossing along one column, linearly interpolated.

    ``P_col`` is ``P`` as a function of ``Y_arr`` at fixed ``R``. Returns the
    interpolated ``eta`` of the first sign change of ``P_col - level``, or
    ``np.nan`` if the column never crosses ``level``.
    """
    d = P_col - level
    for i in range(len(d) - 1):
        a, b = d[i], d[i + 1]
        if a == 0.0:
            return float(Y_arr[i])
        if a * b < 0.0:
            t = a / (a - b)
            return float(Y_arr[i] + t * (Y_arr[i + 1] - Y_arr[i]))
    return float("nan")


def contour_band_width(
    Y_arr: np.ndarray,
    P_lower: np.ndarray,
    P_upper: np.ndarray,
    level: float = 0.5,
) -> dict:
    """Positional uncertainty of the ``P = level`` contour, in ``eta``.

    For each ``R`` column we interpolate where the ``P_lower`` (16th-pct) and
    ``P_upper`` (84th-pct) maps cross ``level``; the absolute difference is the
    68% band width ``Delta eta`` of the contour at that ``R``.  We report the
    median and worst-case (max) width over all columns that cross ``level``.

    Parameters
    ----------
    Y_arr : (ny,) eta grid (rows of the P maps).
    P_lower, P_upper : (ny, nx) 16th/84th-percentile probability maps from
        :meth:`ProbabilisticChart.bootstrap_contour`.
    level : contour probability (default 0.5).

    Returns
    -------
    dict with ``median_width``, ``max_width``, ``n_columns``, ``level``.
    """
    P_lower = np.asarray(P_lower)
    P_upper = np.asarray(P_upper)
    ny, nx = P_lower.shape
    widths = []
    for ix in range(nx):
        y_lo = _column_crossing(P_lower[:, ix], Y_arr, level)
        y_hi = _column_crossing(P_upper[:, ix], Y_arr, level)
        if np.isnan(y_lo) or np.isnan(y_hi):
            continue
        widths.append(abs(y_lo - y_hi))
    widths = np.asarray(widths)
    if widths.size == 0:
        return {"median_width": float("nan"), "max_width": float("nan"),
                "n_columns": 0, "level": level}
    return {
        "median_width": float(np.median(widths)),
        "max_width": float(np.max(widths)),
        "n_columns": int(widths.size),
        "level": level,
    }


def _compute_one_bootstrap_chart(args):
    """ProcessPool worker: compute a single P_grid for one seed.

    Defined at module scope so it pickles cleanly under macOS spawn.
    """
    (eps_ratio, r_max, y_max, steps, N_mc, threshold,
     seed, y_min, budget) = args
    pc = ProbabilisticChart(budget=budget)
    _, _, P = pc.compute_probability_grid(
        eps_ratio=eps_ratio, r_max=r_max, y_max=y_max,
        steps=steps, N_mc=N_mc, threshold=threshold,
        seed=int(seed), y_min=y_min, verbose=False,
    )
    return P


def instability_probability(
    coords: dict,
    *,
    N_mc: int = 200,
    threshold: float = 0.01,
    budget: Optional[PerturbationBudget] = None,
    seed: int = 42,
    l_modes=(2, 3, 4),
    y_min: float = Y_MIN_DEFAULT,
    should_stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Per-cell ``P(gamma/nu_0x > threshold)`` under the jitter budget.

    ``coords`` is the output of
    :func:`linac_gen.analysis.hofmann_stability.hofmann_coordinates` (or of
    ``hofmann_stability``, which carries the same keys).  One
    ``sample_perturbations`` draw + one batched solver call per cell; cells
    with non-finite coordinates return NaN.  A single generator seeded once
    is drawn sequentially across cells, so the result is deterministic for
    a given ``(coords, N_mc, seed)`` — which also means a cell's P depends
    on the WHOLE coords vector, not its own coordinates alone: a NaN cell
    upstream draws nothing and shifts every later cell's samples.
    """
    R = np.asarray(coords["R"], float)
    Y = np.asarray(coords["Y"], float)
    EPS = np.asarray(coords["eps_ratio"], float)
    n = R.size
    out = np.full(n, np.nan)
    if coords.get("reason") is not None:
        return out

    solver = DispersionSolver()
    bud = budget if budget is not None else PerturbationBudget()
    rng = np.random.default_rng(seed)
    for k in range(n):
        if should_stop is not None and should_stop():
            raise ChartAborted("instability-probability computation aborted")
        if not (np.isfinite(R[k]) and np.isfinite(Y[k])
                and np.isfinite(EPS[k]) and EPS[k] > 0):
            continue
        R_p, Y_p, eps_p = sample_perturbations(
            float(R[k]), float(Y[k]), float(EPS[k]), N_mc=N_mc,
            budget=bud, rng=rng, y_min=y_min,
        )
        g = solver.growth_rate_batch(R_p, Y_p, eps_p, l_modes=l_modes)
        out[k] = float((g > threshold).sum()) / N_mc
    return out


__all__ = [
    "PerturbationBudget",
    "ProbabilisticChart",
    "sample_perturbations",
    "instability_probability",
    "contour_band_width",
    "Y_MIN_DEFAULT",
]
