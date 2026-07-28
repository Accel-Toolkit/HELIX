"""Multi-objective lattice design for HELIX.

Where ``match()`` drives the ADJUST knobs to satisfy SET-card residuals
(a single scalar cost), this module explores the **trade-off surface**
between *competing* objectives -- e.g. transverse emittance growth vs.
beam loss vs. exit energy -- and returns a **Pareto front** of
non-dominated designs.  This is the canonical offline accelerator-design
workflow (surrogate / fast-model as the forward pass, a multi-objective
optimiser sweeping the decision box).

Decision variables come from the same ``ADJUST`` cards the matcher uses
(``collect_variables`` + finite bounds).  Objectives are chosen by name
from :data:`OBJECTIVES` -- a small library keyed to the diagnostics the
forward pass already records.  Two optimisers are available:

* ``"nsga2"`` (default) -- NSGA-II genetic algorithm (pymoo).  Robust,
  population-based; the standard accelerator lattice-design tool.  Best
  when the forward pass is cheap (envelope / surrogate).
* ``"qnehvi"`` -- Bayesian multi-objective (BoTorch qNEHVI).  Sample
  efficient -- far fewer forward passes -- so better when each evaluation
  is expensive (``cost_solver="mp"`` / space charge).

All objectives are framed as **minimisation** (maximise-style quantities
are negated in the library), so a smaller value is always better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from linac_gen.core.config import BeamConfig
from linac_gen.matching.variables import collect_variables
from linac_gen.matching.engine import (
    _run_envelope, _run_mp, _apply_x,
    _link_group_index, _build_x0_and_bounds, _require_finite_bounds,
)


# ----------------------------------------------------------------------
# Built-in objective library.  Each entry maps a name -> (extractor,
# label) where extractor(results) -> float (already minimisation-framed).
# These read fields the DiagnosticRecorder / EnvelopeResults populate.
# ----------------------------------------------------------------------
def _last(seq, default=0.0):
    return float(seq[-1]) if seq is not None and len(seq) else float(default)


def _first(seq, default=0.0):
    return float(seq[0]) if seq is not None and len(seq) else float(default)


def _growth(name):
    """Exit/entry ratio of a recorded sequence (minimise -> 1.0 ideal)."""
    def f(r):
        seq = getattr(r, name, None)
        if not seq or len(seq) < 1:
            return 0.0
        a0 = _first(seq, 1.0)
        return _last(seq, 0.0) / a0 if abs(a0) > 1e-30 else _last(seq, 0.0)
    return f


def _emit_n_growth(plane):
    """Normalised-emittance growth ratio for a transverse/long plane,
    deriving emit_n from geometric emit x betagamma when emit_n* is
    absent (envelope mode)."""
    geo = {"x": "emit_x", "y": "emit_y", "z": "emit_z_mmmrad"}[plane]
    norm = {"x": "emit_nx", "y": "emit_ny", "z": "emit_nz"}[plane]

    def f(r):
        en = getattr(r, norm, None)
        if en and len(en) >= 1:
            seq = en
        else:
            g = getattr(r, geo, None)
            beta = getattr(r, "ref_beta", None)
            gam = getattr(r, "ref_gamma", None)
            if (not g or not beta or not gam
                    or len(g) != len(beta) or len(beta) != len(gam)):
                return 0.0
            seq = [gi * bi * gj for gi, bi, gj in zip(g, beta, gam)]
        a0 = _first(seq, 1.0)
        return _last(seq, 0.0) / a0 if abs(a0) > 1e-30 else _last(seq, 0.0)
    return f


OBJECTIVES: dict = {
    "emit_nx_growth": (_emit_n_growth("x"), "norm. emittance growth, x (out/in)"),
    "emit_ny_growth": (_emit_n_growth("y"), "norm. emittance growth, y (out/in)"),
    "emit_nz_growth": (_emit_n_growth("z"), "norm. emittance growth, z (out/in)"),
    "emit_4d_growth": (_growth("emit_4d"), "4-D emittance growth (out/in)"),
    "transmission_loss": (
        lambda r: 100.0 - _last(getattr(r, "transmission", None), 100.0),
        "beam loss [%] (100 - final transmission); MP only",
    ),
    "neg_exit_energy": (
        lambda r: -_last(getattr(r, "ref_w_kin", None), 0.0),
        "negative exit kinetic energy [MeV] (minimise -> maximise W)",
    ),
    "exit_sigma_x": (
        lambda r: _last(getattr(r, "sigma_x", None), 0.0),
        "exit horizontal RMS size [mm]",
    ),
    "exit_sigma_y": (
        lambda r: _last(getattr(r, "sigma_y", None), 0.0),
        "exit vertical RMS size [mm]",
    ),
    "max_sigma_x": (
        lambda r: float(np.max(getattr(r, "sigma_x", [0.0]) or [0.0])),
        "peak horizontal RMS size along s [mm]",
    ),
    "max_sigma_y": (
        lambda r: float(np.max(getattr(r, "sigma_y", [0.0]) or [0.0])),
        "peak vertical RMS size along s [mm]",
    ),
}


def objective_labels(names: List[str]) -> List[str]:
    return [OBJECTIVES[n][1] for n in names]


@dataclass
class ParetoResult:
    """Outcome of a multi-objective design run.

    The ``_x`` arrays have one column per *optimiser column* (``n_cols``),
    which is the link-group-deduplicated decision width: ADJUST variables
    sharing a non-zero ``link_group`` collapse to a single column.
    ``variables`` is the *full* list (one entry per ADJUST DoF), so it can
    be longer than ``pareto_x``/``all_x`` are wide.  Use
    :meth:`column_variables` / :meth:`column_variable_labels` to get the
    per-column representatives that align with the ``_x`` columns (CSV /
    table writers must use those, not ``variables`` directly).
    """
    objective_names: List[str]
    pareto_x: np.ndarray            # (n_front, n_cols) decision vectors
    pareto_F: np.ndarray            # (n_front, n_obj) objective vectors
    all_x: np.ndarray              # (n_eval, n_cols) every point evaluated
    all_F: np.ndarray              # (n_eval, n_obj)
    variables: list = field(default_factory=list)
    col_for_var: list = field(default_factory=list)
    n_cols: int = 0
    algorithm: str = "nsga2"
    n_eval: int = 0
    message: str = ""

    def column_variables(self) -> list:
        """One representative ``Variable`` per optimiser column, in column
        order, aligned to the columns of :attr:`pareto_x` / :attr:`all_x`.

        Linked variables (shared ``link_group``) collapse to one column;
        this returns the first variable encountered for each column.  When
        nothing is linked this is just :attr:`variables` unchanged.
        """
        if not self.variables:
            return []
        cfv = self.col_for_var or list(range(len(self.variables)))
        n = self.n_cols or ((max(cfv) + 1) if cfv else 0)
        reps: list = [None] * n
        for var, col in zip(self.variables, cfv):
            if 0 <= col < n and reps[col] is None:
                reps[col] = var
        return [v for v in reps if v is not None]

    def column_variable_labels(self) -> List[str]:
        """Decision-variable labels aligned to the ``_x`` columns."""
        return [v.label for v in self.column_variables()]


def _validate_objectives(objective_names):
    if not objective_names or len(objective_names) < 2:
        raise ValueError(
            "multi-objective design needs at least 2 objectives; "
            f"got {objective_names!r}.  Available: {sorted(OBJECTIVES)}")
    for n in objective_names:
        if n not in OBJECTIVES:
            raise ValueError(
                f"unknown objective {n!r}; available: {sorted(OBJECTIVES)}")


def _pareto_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of F (minimisation)."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j <= i in all objs and < in at least one.
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def pareto_optimize(
    lattice, beam_cfg: BeamConfig, objective_names: List[str], *,
    algorithm: str = "nsga2",
    space_charge: bool = False,
    cost_solver: str = "envelope",
    mp_n_particles: int = 1000,
    mp_seed: int = 42,
    pop_size: int = 24,
    n_gen: int = 15,
    seed: int = 0,
    callback: Optional[Callable[[int, int], None]] = None,
) -> ParetoResult:
    """Explore the trade-off surface between ``objective_names`` over the
    lattice's ADJUST decision variables.

    .. note::
       ``lattice`` is mutated in place: on return it is left at the first
       Pareto design (best on the first objective) so a subsequent forward
       run reflects a real point on the returned front.  Callers that need
       the original state should pass a copy (the GUI deep-copies).

    Parameters
    ----------
    objective_names :
        >=2 names from :data:`OBJECTIVES`.
    algorithm :
        ``"nsga2"`` (genetic, default) or ``"qnehvi"`` (Bayesian MO).
    pop_size, n_gen :
        NSGA-II population and generation count.  For qnehvi, ``pop_size``
        seeds the initial design and ``n_gen`` is the number of BO
        iterations.
    callback :
        ``callback(done_evals, total_evals_estimate)`` for GUI progress.
    """
    _validate_objectives(objective_names)
    if cost_solver not in ("envelope", "mp"):
        raise ValueError(f"unknown cost_solver {cost_solver!r}")
    # Cross-check the objective/solver pairing: transmission does not
    # exist in envelope results (the OBJECTIVES entry itself says "MP
    # only"), so under the envelope cost solver the objective would sit
    # silently at its 100%-transmission default for every design point —
    # a constant column in the Pareto front (2026-07-25 review, claim 9).
    _MP_ONLY = ("transmission_loss",)
    bad = [n for n in objective_names if n in _MP_ONLY]
    if bad and cost_solver != "mp":
        raise ValueError(
            f"objective(s) {bad!r} require cost_solver='mp' — the envelope "
            "forward pass has no transmission, so the objective would be a "
            "constant and the Pareto front over it meaningless.")

    variables = collect_variables(lattice, beam_cfg)
    if not variables:
        raise ValueError("no ADJUST variables in lattice to optimise")
    col_for_var, n_cols = _link_group_index(variables)
    x0, lo, hi = _build_x0_and_bounds(variables, col_for_var, n_cols)
    _require_finite_bounds("multi-objective", variables, col_for_var, lo, hi)

    extractors = [OBJECTIVES[n][0] for n in objective_names]
    n_obj = len(objective_names)

    all_x: list = []
    all_F: list = []
    state = {"n": 0}

    def _forward(x):
        _apply_x(np.asarray(x, dtype=float), variables, col_for_var)
        if cost_solver == "mp":
            r = _run_mp(lattice, beam_cfg, space_charge=space_charge,
                        n_particles=mp_n_particles, seed=mp_seed)
        else:
            r = _run_envelope(lattice, beam_cfg, space_charge=space_charge)
        F = np.array([float(e(r)) for e in extractors], dtype=float)
        # Guard NaN/inf so the optimiser doesn't choke -- push to a large
        # finite penalty (a failed/unstable design is simply bad).
        F = np.where(np.isfinite(F), F, 1e9)
        all_x.append(np.asarray(x, dtype=float).copy())
        all_F.append(F.copy())
        state["n"] += 1
        if callback is not None:
            callback(state["n"], -1)
        return F

    if algorithm == "nsga2":
        msg = _run_nsga2(_forward, lo, hi, n_obj, pop_size, n_gen, seed)
    elif algorithm == "qnehvi":
        msg = _run_qnehvi(_forward, x0, lo, hi, n_obj, pop_size, n_gen, seed)
    else:
        raise ValueError(
            f"unknown multi-objective algorithm {algorithm!r}; "
            f"choose 'nsga2' or 'qnehvi'")

    X = np.asarray(all_x, dtype=float)
    F = np.asarray(all_F, dtype=float)
    mask = _pareto_mask(F)
    # Sort the front by the first objective for a tidy, plottable order.
    front_idx = np.where(mask)[0]
    front_idx = front_idx[np.argsort(F[front_idx, 0])]

    # Leave the lattice at the first (best-on-obj-0) Pareto design so a
    # subsequent forward run reflects a real point on the front.
    if front_idx.size:
        _apply_x(X[front_idx[0]], variables, col_for_var)

    return ParetoResult(
        objective_names=list(objective_names),
        pareto_x=X[front_idx], pareto_F=F[front_idx],
        all_x=X, all_F=F, variables=variables,
        col_for_var=list(col_for_var), n_cols=int(n_cols),
        algorithm=algorithm, n_eval=int(state["n"]), message=msg,
    )


def _run_nsga2(forward, lo, hi, n_obj, pop_size, n_gen, seed):
    """NSGA-II via pymoo."""
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    d = lo.size

    class _Prob(Problem):
        def __init__(self):
            super().__init__(n_var=d, n_obj=n_obj,
                             xl=np.asarray(lo, dtype=float),
                             xu=np.asarray(hi, dtype=float))

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.vstack([forward(row) for row in np.atleast_2d(X)])

    algo = NSGA2(pop_size=int(pop_size))
    pymoo_minimize(
        _Prob(), algo, ("n_gen", int(n_gen)),
        seed=int(seed), verbose=False,
    )
    return f"NSGA-II: pop {pop_size} x {n_gen} gen"


def _run_qnehvi(forward, x0, lo, hi, n_obj, n_init, n_iter, seed):
    """Bayesian multi-objective via BoTorch qNEHVI."""
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.optim import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (  # noqa: E501
            FastNondominatedPartitioning,
        )
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # noqa: F841
        raise ImportError(
            "the 'qnehvi' multi-objective algorithm requires botorch + "
            "gpytorch; install with: pip install botorch gpytorch")

    torch.manual_seed(int(seed))
    dtype = torch.double
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.size
    span = np.where(hi > lo, hi - lo, 1.0)
    unit_bounds = torch.stack([torch.zeros(d, dtype=dtype),
                               torch.ones(d, dtype=dtype)])

    def to_unit(x):
        return np.clip((np.asarray(x, float) - lo) / span, 0.0, 1.0)

    def from_unit(u):
        return lo + np.asarray(u, float) * span

    X_u, Y = [], []

    def ev(x_orig):
        # qNEHVI MAXIMISES hypervolume of MAXIMISATION objectives; we
        # minimise, so store -F.
        F = forward(x_orig)
        X_u.append(to_unit(x_orig))
        Y.append([-float(v) for v in F])

    ev(np.asarray(x0, float))
    sob = draw_sobol_samples(bounds=unit_bounds, n=int(max(n_init, 2 * d)),
                             q=1, seed=int(seed) + 1).squeeze(1).cpu().numpy()
    for u in sob:
        ev(from_unit(u))

    for _ in range(int(n_iter)):
        tx = torch.tensor(np.asarray(X_u), dtype=dtype)
        ty = torch.tensor(np.asarray(Y), dtype=dtype)
        try:
            models = [
                SingleTaskGP(tx, ty[:, i:i + 1],
                             outcome_transform=Standardize(m=1))
                for i in range(n_obj)
            ]
            model = ModelListGP(*models)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            ref_point = ty.min(dim=0).values - 0.1 * ty.std(dim=0).clamp(min=1e-6)
            part = FastNondominatedPartitioning(ref_point=ref_point, Y=ty)
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref_point.tolist(),
                X_baseline=tx, partitioning=part, prune_baseline=True,
            )
            cand, _ = optimize_acqf(
                acqf, bounds=unit_bounds, q=1,
                num_restarts=8, raw_samples=128,
            )
            u_next = cand.squeeze(0).detach().cpu().numpy()
        except Exception:  # noqa: BLE001 -- GP/acqf hiccup -> Sobol probe
            u_next = draw_sobol_samples(
                bounds=unit_bounds, n=1, q=1,
                seed=int(seed) + 500 + len(X_u),
            ).squeeze(1).squeeze(0).cpu().numpy()
        ev(from_unit(u_next))

    # Report the *actual* initial-design size: the x0 seed plus the Sobol
    # batch of max(n_init, 2*d) points (not the bare n_init/pop_size arg).
    return f"qNEHVI: {len(sob) + 1} init + {n_iter} BO iters"
