"""Bayesian-optimisation driver for the HELIX matcher.

This is the engine behind ``match(..., algorithm="bayesopt")``.  It is a
thin, self-contained wrapper around BoTorch's single-task Gaussian-process
+ acquisition loop that plugs into the matcher's existing
``cost_fn(x) -> float`` seam:

* The optimiser is **black-box**: it only ever calls ``cost_fn`` (the same
  scalar objective CMA-ES / DE / dual-annealing use), so iteration
  counting, the progress ``callback``, best-x tracking, and the
  ``StopIteration -> _MatchCancelled`` cancellation contract all keep
  working unchanged -- they live inside ``cost_fn``/``fn`` in engine.py.
* It works in the **unit cube**: parameters are normalised to [0, 1]^d
  via the finite ``(lo, hi)`` bounds the matcher already requires for the
  global algorithms, then de-normalised before each ``cost_fn`` call.

Why BO: Gaussian-process Bayesian optimisation is the accelerator-tuning
state of the art (SLAC Xopt / Badger).  It targets the matcher's
worst case -- **expensive** forward passes (``cost_solver="mp"`` /
space-charge matches) and **multimodal / one-sided** landscapes
(coupling resonances, sync-phase sign flips, ``MIN_EMIT_GROWTH`` plateaus)
-- reaching the optimum in far fewer evaluations than the
population-based globals.  It is NOT a default: for cheap linear-envelope
matches, ``least_squares`` / ``gradient`` win on wall-clock.

Physics-informed warm start (``prior_cost_fn``): when the caller supplies
a *cheap* objective (HELIX's differentiable envelope cost, used as a
stand-in for the expensive MP cost), the driver evaluates it on a small
Sobol batch and seeds the initial design with the cheapest-physics points.
This concentrates the GP's first samples near the envelope-optimal region
so the expensive objective is queried where it is most likely to pay off
-- the in-box analogue of the "prior-mean model" technique (PRAB 27
084801, 2024).  Honest scope: this is warm-starting, not a learned GP
mean module; it still cuts expensive evaluations when envelope tracks MP.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class _Res:
    """Duck-typed stand-in for a scipy OptimizeResult, matching what the
    matcher's common tail reads (``x`` / ``success`` / ``message``)."""

    __slots__ = ("x", "success", "message", "nit")

    def __init__(self, x, success, message, nit=0):
        self.x = x
        self.success = bool(success)
        self.message = str(message)
        self.nit = int(nit)


def run_bayesopt(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_iter: int = 60,
    n_init: int = 0,
    seed: int = 0,
    prior_cost_fn: Optional[Callable[[np.ndarray], float]] = None,
    n_prior: int = 8,
) -> _Res:
    """Minimise ``cost_fn`` over the box ``[lo, hi]`` with Bayesian
    optimisation.

    Parameters
    ----------
    cost_fn :
        Scalar objective ``cost_fn(x) -> float`` (the matcher's ``cost_fn``).
        May raise ``_MatchCancelled`` (via the GUI Stop button) -- that
        propagates out of here unchanged for the engine branch to catch.
    x0, lo, hi :
        Initial point and finite lower/upper bounds (1-D arrays, length d).
    max_iter :
        Number of *Bayesian* iterations (sequential, one new point each).
        This is the expensive-evaluation budget that matters.
    n_init :
        Number of initial space-filling (Sobol) samples before the GP is
        first fit.  ``0`` -> a sensible default ``min(2*d + 2, 12)``.
    seed :
        Seed for Sobol + torch (reproducible runs).
    prior_cost_fn :
        Optional *cheap* objective for physics-informed warm-starting.
        When supplied, ``n_prior`` Sobol points are scored with it and the
        cheapest are folded into the initial design.
    n_prior :
        Size of the cheap-physics warm-start Sobol batch.

    Returns
    -------
    _Res with ``x`` = best point found (original units), ``success``,
    ``message``.
    """
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # noqa: F841
        raise ImportError(
            "the 'bayesopt' algorithm requires botorch + gpytorch; "
            "install with: pip install botorch gpytorch"
        )

    torch.manual_seed(int(seed))
    dtype = torch.double
    x0 = np.asarray(x0, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = x0.size
    span = np.where(hi > lo, hi - lo, 1.0)

    def to_unit(x):
        return np.clip((np.asarray(x, dtype=float) - lo) / span, 0.0, 1.0)

    def from_unit(u):
        return lo + np.asarray(u, dtype=float) * span

    # Unit-cube bounds tensor for the acquisition optimiser.
    unit_bounds = torch.stack([
        torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)
    ])

    # We MINIMISE cost; BoTorch MAXIMISES.  Track Y = -cost.
    X_unit: list[np.ndarray] = []
    Y_neg: list[float] = []
    best_cost = float("inf")
    best_x = np.asarray(x0, dtype=float).copy()

    def _eval(x_orig):
        nonlocal best_cost, best_x
        c = float(cost_fn(np.asarray(x_orig, dtype=float)))
        X_unit.append(to_unit(x_orig))
        Y_neg.append(-c)
        if c < best_cost:
            best_cost = c
            best_x = np.asarray(x_orig, dtype=float).copy()
        return c

    # ---- initial design -------------------------------------------------
    if n_init <= 0:
        n_init = min(2 * d + 2, 12)

    # Always include x0 -- gives the GP the unmatched-lattice anchor and
    # guarantees BO can't do worse than the starting point.
    _eval(x0)

    init_pts: list[np.ndarray] = []

    # Physics-informed warm start: score a cheap-physics Sobol batch and
    # take the best few as seeds.
    if prior_cost_fn is not None and n_prior > 0:
        sob = draw_sobol_samples(
            bounds=unit_bounds, n=int(n_prior), q=1, seed=int(seed) + 1
        ).squeeze(1).cpu().numpy()
        scored = []
        for u in sob:
            try:
                pc = float(prior_cost_fn(from_unit(u)))
            except Exception:  # noqa: BLE001 -- a flaky cheap eval must not kill BO
                continue
            if np.isfinite(pc):
                scored.append((pc, u))
        scored.sort(key=lambda t: t[0])
        keep = max(1, min(len(scored), (n_init + 1) // 2))
        init_pts.extend(u for _pc, u in scored[:keep])

    # Fill the remaining initial design with plain Sobol exploration.
    n_remaining = max(0, n_init - len(init_pts))
    if n_remaining > 0:
        sob2 = draw_sobol_samples(
            bounds=unit_bounds, n=int(n_remaining), q=1, seed=int(seed) + 2
        ).squeeze(1).cpu().numpy()
        init_pts.extend(sob2)

    for u in init_pts:
        _eval(from_unit(u))

    # ---- Bayesian loop --------------------------------------------------
    n_bo = max(1, int(max_iter))
    for _ in range(n_bo):
        train_x = torch.tensor(np.asarray(X_unit), dtype=dtype)
        train_y = torch.tensor(np.asarray(Y_neg), dtype=dtype).unsqueeze(-1)
        # Fit the GP.  A fit hiccup (singular kernel on duplicate points,
        # etc.) must not abort the match -- fall back to a Sobol probe.
        try:
            gp = SingleTaskGP(
                train_x, train_y,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = qLogExpectedImprovement(
                model=gp, best_f=train_y.max(),
            )
            cand, _ = optimize_acqf(
                acqf, bounds=unit_bounds, q=1,
                num_restarts=10, raw_samples=256,
            )
            u_next = cand.squeeze(0).detach().cpu().numpy()
        except Exception:  # noqa: BLE001
            u_next = draw_sobol_samples(
                bounds=unit_bounds, n=1, q=1,
                seed=int(seed) + 100 + len(X_unit),
            ).squeeze(1).squeeze(0).cpu().numpy()
        _eval(from_unit(u_next))

    msg = (f"Bayesian optimisation: {len(X_unit)} evals, "
           f"best cost {best_cost:.4e}"
           + ("  (physics-informed warm start)"
              if prior_cost_fn is not None else ""))
    return _Res(x=best_x, success=True, message=msg, nit=len(X_unit))
