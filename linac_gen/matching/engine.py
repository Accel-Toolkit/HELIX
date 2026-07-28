"""Lattice-driven matcher: ``ADJUST`` + ``SET`` cards → scipy.least_squares.

The :func:`match` function is the package-level entry point.  Pass it a
:class:`linac_gen.core.lattice.Lattice` containing ``LatticeCommand``
elements plus a :class:`linac_gen.core.config.BeamConfig`; it builds
:class:`Variable`\\s and :class:`Constraint`\\s, runs Levenberg–Marquardt
(``method='trf'``, bounds-aware) until residuals converge or the
iteration cap is hit, and returns a :class:`MatchResult`.

Variables and the input ``BeamConfig`` are mutated **in place**: after a
successful run, the lattice's element parameters and the beam's Twiss /
emittance / centroid attributes carry the matched values.

The engine runs an envelope simulation per residual evaluation
(``EnvelopeSolver``).  Space-charge mode is opt-in via
``space_charge=True``; otherwise SC is disabled (``current=0.0``) so
each iteration is fast.
"""
from __future__ import annotations

import copy
import math
import time


class _MatchCancelled(Exception):
    """Internal signal: the user clicked Stop in the GUI (the worker's
    callback raised StopIteration).  We convert it to a non-StopIteration
    exception inside fn() / fn_grad() because some scipy optimisers
    (notably ``differential_evolution``'s batched evaluator) silently
    swallow ``StopIteration`` via PEP-479 / internal generator wrappers,
    surfacing as a misleading "func must return scalar" RuntimeError.
    Each algorithm branch catches ``_MatchCancelled`` and synthesizes a
    cancelled MatchResult.  Outside this module nobody sees the exception.
    """
    pass
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from linac_gen.core.config import BeamConfig
from linac_gen.matching.constraints import Constraint, collect_constraints
from linac_gen.matching.variables import (MatchingConfigError, Variable,
                                          collect_variables)

#: Optimisers :func:`match` can dispatch to.  ``least_squares`` and
#: ``gradient`` are local; ``differential_evolution``, ``dual_annealing``
#: and ``cmaes`` are global and need finite variable bounds.
MATCH_ALGORITHMS = ("least_squares", "differential_evolution",
                    "dual_annealing", "gradient", "cmaes",
                    "sequential_scan", "bayesopt")


@dataclass
class MatchResult:
    """Outcome of a :func:`match` run.

    Attributes
    ----------
    success :
        True when the optimiser reports convergence *or* the residual cost
        is below 1e-10 (we accept "essentially zero" residuals as
        success regardless of the optimiser's internal status).
    message :
        Optimiser termination message verbatim.
    n_iter :
        Number of cost-function evaluations.
    elapsed_s :
        Wall time spent in :func:`match`.
    x0 :
        Initial variable column.
    x_final :
        Final variable column.  Already written into the targets.
    residuals :
        Final residual vector (post-weight).
    cost :
        ``0.5 * Σ residuals²``.
    variables :
        ``Variable`` objects in collection order — the *full* list (one
        entry per ADJUST DoF).  Linked variables (shared ``link_group``)
        map to the same optimiser column, so this list can be longer than
        ``x0``/``x_final``, which are indexed per optimiser column.
    constraints :
        ``Constraint`` objects in residual order.
    per_constraint_residuals :
        Dict mapping each constraint label to its sub-vector of the
        final residual (handy for reports).
    """

    success: bool
    message: str
    n_iter: int
    elapsed_s: float
    x0: np.ndarray
    x_final: np.ndarray
    residuals: np.ndarray
    cost: float
    variables: List[Variable]
    constraints: List[Constraint]
    per_constraint_residuals: dict = field(default_factory=dict)
    # Initial cost evaluated at x0 with the chosen cost_solver, BEFORE
    # any optimisation step ran.  Populated by match()'s baseline pass
    # (engine.py).  ``cost - baseline_cost`` quantifies the matcher's
    # improvement over the unmatched lattice.  None if the engine did
    # not run a baseline (e.g. cancellation before first eval).
    baseline_cost: Optional[float] = None

    @property
    def z_penalty_infeasible(self) -> bool:
        """True when the reported solution's SET_TWISS longitudinal
        residual slots carry the fixed degeneracy penalty — i.e. the
        match ran under ``allow_inert_constraints`` (or a trial ended
        degenerate) and its 'success' does NOT include the requested
        kaz/kbz targets.  Machine-readable so batch scripts need not
        parse warnings."""
        from linac_gen.matching.constraints import _Z_DEGENERATE_PENALTY
        # per_constraint_residuals stores POST-weight values, so the
        # comparison must scale by each card's weight (SET_TWISS is
        # built with weight=1.0 today, but this must not silently
        # break if that ever changes — adversarial-review finding).
        weights: dict = {}
        for c in (self.constraints or []):
            weights.setdefault(str(getattr(c, "label", "")),
                               float(getattr(c, "weight", 1.0)))
        for label, arr in (self.per_constraint_residuals or {}).items():
            label = str(label)
            if not label.startswith("SET_TWISS"):
                continue
            w = weights.get(label)
            if w is None:       # suffixed duplicate labels
                cands = [v for k, v in weights.items()
                         if k and label.startswith(k)]
                w = cands[0] if cands else 1.0
            a = np.asarray(arr, dtype=float)
            if a.size and np.any(
                    np.isclose(a, _Z_DEGENERATE_PENALTY * w, rtol=1e-12)):
                return True
        return False

    def report(self) -> str:
        """Human-readable summary suitable for CLI / log output."""
        lines = [
            "Matching report",
            "═══════════════",
            f"  status     : {'OK' if self.success else 'FAILED'}"
            + ("  [INFEASIBLE: SET_TWISS z targets unmet — penalty "
               "slots active]" if self.z_penalty_infeasible else ""),
            f"  message    : {self.message}",
            f"  iterations : {self.n_iter}",
            f"  cost       : {self.cost:.6e}",
            f"  elapsed    : {self.elapsed_s:.2f} s",
            "",
            "Variables:",
        ]
        # Map each variable to its optimiser COLUMN — linked variables
        # share one column, so ``variables`` can be longer than
        # ``x0``/``x_final`` and a positional zip mislabels every entry
        # after the first linked pair (and drops the tail).
        col_for_var, _n_cols = _link_group_index(self.variables)
        for var, col in zip(self.variables, col_for_var):
            x0 = self.x0[col] if col < len(self.x0) else float("nan")
            x = (self.x_final[col] if col < len(self.x_final)
                 else float("nan"))
            lines.append(
                f"  {var.label:<32s}  {x0:>12.6g}  →  {x:<12.6g}  "
                f"[{var.vmin:>11.4g}, {var.vmax:<11.4g}]"
            )
        lines.append("")
        lines.append("Constraints:")
        _seen: dict = {}
        for c in self.constraints:
            n = _seen.get(c.label, 0)
            _seen[c.label] = n + 1
            key = c.label if n == 0 else f"{c.label} ({n + 1})"
            res = self.per_constraint_residuals.get(key,
                                                    np.array([0.0]))
            rms = float(np.sqrt(np.mean(res * res))) if res.size else 0.0
            extra = f"  ({c.notes})" if c.notes else ""
            lines.append(
                f"  {c.label:<32s}  residual rms = {rms:.4e}{extra}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _link_group_index(variables: List[Variable]) -> tuple[list[int], int]:
    """Map each variable to its optimiser-column index.

    Variables with the same non-zero ``link_group`` collapse to one
    column; ``link_group == 0`` always gets its own column.
    """
    column_for_var: list[int] = []
    group_to_col: dict[int, int] = {}
    next_col = 0
    for var in variables:
        if var.link_group:
            col = group_to_col.get(var.link_group)
            if col is None:
                col = next_col
                group_to_col[var.link_group] = col
                next_col += 1
            column_for_var.append(col)
        else:
            column_for_var.append(next_col)
            next_col += 1
    return column_for_var, next_col


def _build_x0_and_bounds(variables: List[Variable], col_for_var: list[int],
                         n_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.zeros(n_cols, dtype=float)
    lo = np.full(n_cols, -math.inf, dtype=float)
    hi = np.full(n_cols, math.inf, dtype=float)
    for var, col in zip(variables, col_for_var):
        x0[col] = var.x0
        # Tighten bounds (intersection of all linked-variable bounds).
        lo[col] = max(lo[col], var.vmin)
        hi[col] = min(hi[col], var.vmax)
    # Final clip
    x0 = np.minimum(np.maximum(x0, lo), hi)
    return x0, lo, hi


def _require_finite_bounds(algorithm: str, variables: List[Variable],
                           col_for_var: list[int],
                           lo: np.ndarray, hi: np.ndarray) -> None:
    """Reject open-ended bounds for the global optimisers.

    ``differential_evolution`` and ``dual_annealing`` search a finite box;
    an infinite ``vmin``/``vmax`` (an ADJUST card with no min/max) makes the
    search ill-defined.  Raise a clear, variable-named error rather than
    letting scipy fail obscurely.  ``least_squares`` is unaffected — its TRF
    method handles ±inf bounds natively.
    """
    bad: list[str] = []
    for col in range(len(lo)):
        lo_ok = math.isfinite(lo[col])
        hi_ok = math.isfinite(hi[col])
        if lo_ok and hi_ok:
            continue
        label = next((v.label for v, c in zip(variables, col_for_var)
                      if c == col), f"column {col}")
        sides = ([] if lo_ok else ["lower"]) + ([] if hi_ok else ["upper"])
        bad.append(f"'{label}' ({' and '.join(sides)} bound)")
    if bad:
        raise ValueError(
            f"the '{algorithm}' algorithm needs finite bounds on every "
            f"ADJUST variable; unbounded: {', '.join(bad)}. Add min/max to "
            f"the ADJUST card(s), or use the 'least_squares' algorithm."
        )


def _apply_x(x: np.ndarray, variables: List[Variable],
             col_for_var: list[int]) -> None:
    for var, col in zip(variables, col_for_var):
        var.assign(float(x[col]))


def _run_envelope(lattice, beam_cfg: BeamConfig,
                  *, space_charge: bool):
    """Build a forward-simulation result from the current lattice / beam.

    Imports kept local because ``EnvelopeSolver`` and ``create_beam`` pull
    in matplotlib via downstream modules; we avoid that at package import
    time.
    """
    from linac_gen.distributions.factory import create_beam
    from linac_gen.tracking.envelope import EnvelopeSolver

    beam = create_beam(beam_cfg, seed=42)
    from linac_gen.distributions.factory import geometric_emittances
    _ex, _ey, _ez = geometric_emittances(beam_cfg,
                                         max(beam.ref.bg, 1e-9))
    initial = dict(
        alpha_x=beam_cfg.alpha_x, beta_x=beam_cfg.beta_x, emit_x=_ex,
        alpha_y=beam_cfg.alpha_y, beta_y=beam_cfg.beta_y, emit_y=_ey,
        alpha_z=beam_cfg.alpha_z, beta_z=beam_cfg.beta_z, emit_z=_ez,
        # DC beams must reach the solver: without these keys the
        # envelope cost would run the 3-D bunched-ellipsoid SC kick on
        # a continuous beam while the MP cost honours DC — same
        # matching problem, different physics (mirrors
        # cli/common._envelope_initial).
        continuous=bool(getattr(beam_cfg, "continuous", False)),
        dc_energy_spread_keV=float(
            getattr(beam_cfg, "dc_energy_spread_keV", 0.0)),
        # First-moment seed: envelope results carry the centroid, so
        # envelope-cost matches must launch the same orbit MP would.
        centroid=[float(getattr(beam_cfg, "centroid_x", 0.0) or 0.0),
                  float(getattr(beam_cfg, "centroid_xp", 0.0) or 0.0),
                  float(getattr(beam_cfg, "centroid_y", 0.0) or 0.0),
                  float(getattr(beam_cfg, "centroid_yp", 0.0) or 0.0),
                  float(getattr(beam_cfg, "centroid_dphi", 0.0) or 0.0),
                  float(getattr(beam_cfg, "centroid_dw", 0.0) or 0.0)],
    )
    current = float(beam_cfg.current) if space_charge else 0.0
    solver = EnvelopeSolver(lattice, beam.ref, initial, current=current)
    return solver.run()


def _run_mp(lattice, beam_cfg: BeamConfig, *, space_charge: bool,
            n_particles: int, seed: int,
            sc_config=None, step_config=None):
    """Build an MP forward-simulation result for the matcher's cost
    function.  Returns a ``DiagnosticRecorder`` whose ``emit_x``,
    ``emit_y``, ``emit_z``, ``emit_4d``, ``ref_beta``, ``ref_gamma``,
    ``ref_w_kin`` arrays are populated identically to the envelope
    path so the existing constraint evaluators consume it transparently.

    Thin wrapper around ``linac_gen.cli.common.run_mp_sim`` to avoid
    duplicating the MP tracker setup.  Defaults for ``sc_config`` and
    ``step_config`` come from ``BeamConfig`` defaults; the GUI worker
    constructs both from the Convergence tab and passes them in.
    """
    # Local imports (matches _run_envelope's hygiene -- avoid pulling
    # matplotlib / torch at engine import time).
    from copy import copy
    from linac_gen.cli.common import run_mp_sim
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.step_config import StepConfig

    # Caller may pass None for either config -- use the BeamConfig's
    # current to decide whether SC should be active, mirroring the
    # envelope path's space_charge boolean.  In the matcher, SC-ness is
    # ALWAYS an explicit choice (the engine's space_charge flag / the
    # mo CLI's --no-space-charge / the GUI checkbox), so "no SC" maps
    # to the "off" sentinel -- same physics as None, but it declares
    # intent and suppresses NoSpaceChargeWarning on every evaluation.
    if sc_config is None:
        sc_config = (SpaceChargeConfig()
                     if space_charge and float(beam_cfg.current) > 0
                     else "off")
    elif not space_charge:
        sc_config = "off"
    if step_config is None:
        step_config = StepConfig()
    # Override the beam-config's particle count for this match.
    # ``create_beam`` reads ``cfg.n_particles``; make a temporary
    # shallow copy so we don't mutate the caller's BeamConfig.
    beam_cfg_for_match = copy(beam_cfg)
    beam_cfg_for_match.n_particles = int(max(1, n_particles))
    recorder, _beam = run_mp_sim(
        lattice, beam_cfg_for_match, sc_config, step_config,
        seed=int(seed),
    )
    return recorder


# Module-level globals for the multiprocessing CMA-ES Pool workers.
# Set by ``_cmaes_pool_init`` once per worker process; never touched
# from the main process.  This keeps the per-eval pickle small (just
# the candidate x), since the lattice / beam / variables are shared
# via worker-local state and don't need to traverse the pipe.
_PARALLEL_STATE: dict = {}


def _cmaes_pool_init(lattice, beam_cfg, variables, col_for_var,
                     space_charge: bool,
                     search_solver: str = "envelope",
                     mp_kwargs: dict | None = None) -> None:
    """Pool ``initializer`` -- stash per-worker state.

    Worker processes inherit a copy of these via fork or rebuild via
    spawn; either way the heavy objects (Lattice, BeamConfig, the
    constraint list) need to be unpickled at most once per worker
    lifetime, not once per task.

    ``search_solver`` selects the objective the workers score
    ("envelope" or "mp") — an explicit caller choice since the honesty
    round; workers no longer hardwire the envelope cost.
    """
    import os as _os
    from linac_gen.matching.constraints import collect_constraints
    # N workers × threaded FFTs oversubscribes the box — same guard as
    # parallel/scan_pool.
    _os.environ.setdefault("LINAC_GEN_FFT_WORKERS", "1")
    _PARALLEL_STATE["lattice"] = lattice
    _PARALLEL_STATE["beam_cfg"] = beam_cfg
    _PARALLEL_STATE["variables"] = variables
    _PARALLEL_STATE["col_for_var"] = col_for_var
    _PARALLEL_STATE["space_charge"] = space_charge
    _PARALLEL_STATE["search_solver"] = str(search_solver)
    _PARALLEL_STATE["mp_kwargs"] = dict(mp_kwargs or {})
    # Constraints contain closure-bound evaluators; collect them fresh
    # in each worker rather than trying to pickle them across.
    _PARALLEL_STATE["constraints"] = collect_constraints(lattice)


def _cmaes_pool_eval(x_list) -> float:
    """Worker-side: apply ``x``, run the SELECTED forward pass, return
    the scalar cost."""
    import numpy as _np
    state = _PARALLEL_STATE
    x = _np.asarray(x_list, dtype=float)
    _apply_x(x, state["variables"], state["col_for_var"])
    if state.get("search_solver") == "mp":
        results = _run_mp(
            state["lattice"], state["beam_cfg"],
            space_charge=state["space_charge"],
            **state["mp_kwargs"])
    else:
        results = _run_envelope(
            state["lattice"], state["beam_cfg"],
            space_charge=state["space_charge"])
    residuals, _ = _evaluate_residuals(
        state["constraints"], results, state["lattice"])
    return 0.5 * float(residuals @ residuals)


def _evaluate_residuals(constraints: List[Constraint], results, lattice
                        ) -> tuple[np.ndarray, dict]:
    chunks: list[np.ndarray] = []
    per: dict[str, np.ndarray] = {}
    for c in constraints:
        try:
            r = c.evaluate(results, lattice)
        except Exception as exc:    # noqa: BLE001
            # A constraint that raises is treated as "infeasible" so the
            # optimiser steers away.  But silently turning it into 1e6
            # makes a buggy constraint indistinguishable from a genuinely
            # bad match -- log to stderr with the label + exception so
            # the user can tell apart "constraint at its bound" from
            # "constraint crashed".
            import sys
            print(
                f"[match] constraint {c.label!r} raised "
                f"{type(exc).__name__}: {exc} -- residual pinned to 1e6",
                file=sys.stderr,
            )
            r = np.array([1e6])  # unstable / errored → push optimiser away
        # Disambiguate duplicate labels (two SET_SIZE cards share one) so
        # the per-constraint dict doesn't silently collapse them — the
        # report would show the LAST card's residual for both.
        label = c.label
        if label in per:
            n = 2
            while f"{label} ({n})" in per:
                n += 1
            label = f"{label} ({n})"
        per[label] = r
        chunks.append(r)
    if not chunks:
        return np.array([0.0]), per
    return np.concatenate(chunks), per


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def match(lattice, beam_cfg: BeamConfig, *,
          space_charge: bool = False,
          algorithm: str = "least_squares",
          max_iter: int = 200,
          xtol: float = 1e-8,
          ftol: float = 1e-8,
          cmaes_sigma: float = 0.2,
          cmaes_popsize: int = 0,
          cmaes_parallel: int = 1,
          cmaes_search_solver: str = "auto",
          refine: bool = True,
          bo_prior: bool = False,
          bo_n_init: int = 0,
          seqscan_element_names: Optional[List[str]] = None,
          seqscan_passes: int = 2,
          seqscan_steps: int = 7,
          seqscan_step_frac: float = 0.10,
          seqscan_reversal: str = "both_grew",
          seqscan_threshold: str = "input",
          seqscan_reject_loss: bool = False,
          seqscan_loss_threshold_pct: float = 100.0,
          cost_solver: str = "envelope",
          mp_n_particles: Optional[int] = None,
          mp_seed: int = 42,
          mp_sc_config: Optional[Any] = None,
          mp_step_config: Optional[Any] = None,
          callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
          allow_inert_constraints: bool = False,
          ) -> MatchResult:
    """Run the matcher on a lattice + beam config.

    Parameters
    ----------
    lattice :
        Source lattice.  Mutated in place.
    beam_cfg :
        Initial beam configuration; ``ADJUST_BEAM_*`` mutations land here.
    space_charge :
        When ``True`` the envelope forward pass runs at the configured
        ``beam_cfg.current``; otherwise SC is disabled (faster, less
        accurate for high-current matching).
    algorithm :
        Optimiser to dispatch to — one of :data:`MATCH_ALGORITHMS`:

        * ``"least_squares"`` (default) — local, gradient-based
          trust-region-reflective least squares
          (``scipy.optimize.least_squares`` with
          ``method='trf'``).  Fast; accepts open-ended bounds.
        * ``"differential_evolution"`` — global, gradient-free
          (``scipy.optimize.differential_evolution``).
        * ``"dual_annealing"`` — global simulated annealing
          (``scipy.optimize.dual_annealing``).
        * ``"gradient"`` — local Levenberg–Marquardt driven by an exact
          autograd Jacobian from differentiable matrix tracking.
          Supports SET_TWISS / SET_SIZE matching of quad / solenoid /
          dipole variables, without space charge; raises a clear
          :class:`ValueError` for anything outside that subset.
        * ``"cmaes"`` — Covariance Matrix Adaptation Evolution Strategy
          (Hansen).  Robust global, gradient-free, handles bounds
          natively; the default choice for multimodal landscapes
          (coupling resonances, synchronous-phase sign flips) and the
          one-sided ``MIN_EMIT_GROWTH`` / ``SET_KE_OUT_MIN`` constraints
          whose zero-gradient regions defeat scipy LM.  Needs finite
          variable bounds and the ``cma`` package (``pip install cma``).

        The three global algorithms (``differential_evolution``,
        ``dual_annealing``, ``cmaes``) need *finite* bounds on every
        ADJUST variable; a :class:`ValueError` naming the offending
        variable is raised otherwise.  They run deterministically (fixed
        ``rng`` / ``seed``).
    max_iter :
        For ``least_squares`` the cap on cost-function evaluations; for
        the global algorithms their ``maxiter`` (generations / annealing
        iterations).
    xtol, ftol :
        Tolerances forwarded to ``scipy.least_squares``.  The global
        algorithms use their own scipy-default convergence criteria.
    cmaes_sigma :
        CMA-ES initial step-size as a fraction of the bound-box mean
        width.  Larger values explore more aggressively; default 0.2.
        Ignored unless ``algorithm == "cmaes"``.
    cmaes_popsize :
        CMA-ES population size.  ``0`` (default) means the package
        default ``4 + floor(3·ln(N))``.  Larger populations are more
        robust against local minima at the cost of more forward passes.
    cmaes_search_solver :
        Objective the PARALLEL CMA-ES workers score the population
        search with.  ``"auto"`` (default) follows ``cost_solver``
        faithfully — with ``cost_solver="mp"`` the workers genuinely
        evaluate the multiparticle objective.  ``"envelope"`` is the
        fast envelope-guided hybrid (envelope-scored search; the
        baseline, LS polish and final reported residuals still use the
        requested cost solver) as an explicit opt-in — this was the
        SILENT behavior before 2026-07-19.  ``"mp"`` may be passed for
        explicitness (requires ``cost_solver="mp"`` — validated
        regardless of algorithm or pool size).  Has an effect only
        when the CMA-ES pool is active (``cmaes_parallel != 1``);
        sequential evaluation always uses the requested cost solver.
    refine :
        When ``algorithm == "cmaes"`` (and convergence is non-trivial),
        polish the best CMA-ES point with a short ``least_squares`` run
        before reporting.  Default ``True``.  Has no effect for other
        algorithms.
    callback :
        Optional ``callback(iter_idx, x, cost)`` invoked once per
        residual evaluation.  Useful for GUI progress.  A 4-argument
        callback also receives an ``info`` dict with per-eval scalars
        and ``info["results"]`` — the full forward-sim result of that
        evaluation (READ-ONLY: a fresh object per eval, never mutated
        after the call; powers the GUI's live match preview).
    allow_inert_constraints :
        By default (``False``) the pre-run constraint audit raises if
        the match would silently ignore an active constraint: a stub
        card (parsed for round-trip, evaluator not implemented) with a
        nonzero weight, or ``MIN_TRANSMISSION`` under the envelope
        cost solver (which tracks no particle loss).  Set ``True`` to
        restore the pre-2026-07 warn-and-continue behaviour.
    """
    from scipy.optimize import (
        differential_evolution, dual_annealing, least_squares,
    )

    if algorithm not in MATCH_ALGORITHMS:
        raise ValueError(
            f"unknown matcher algorithm {algorithm!r}; "
            f"choose one of {MATCH_ALGORITHMS}"
        )

    t0 = time.time()
    variables = collect_variables(lattice, beam_cfg)
    constraints = collect_constraints(lattice)

    if not variables:
        # Nothing to vary — return an "already matched" result so the GUI
        # gets a clean empty report instead of an exception.
        results = _run_envelope(lattice, beam_cfg, space_charge=space_charge)
        residuals, per = _evaluate_residuals(constraints, results, lattice)
        return MatchResult(
            success=True, message="No ADJUST variables in lattice",
            n_iter=0, elapsed_s=time.time() - t0,
            x0=np.array([]), x_final=np.array([]),
            residuals=residuals,
            cost=0.5 * float(residuals @ residuals),
            variables=variables, constraints=constraints,
            per_constraint_residuals=per,
        )
    if not constraints:
        return MatchResult(
            success=False, message="No SET constraints in lattice",
            n_iter=0, elapsed_s=time.time() - t0,
            x0=np.array([v.x0 for v in variables]),
            x_final=np.array([v.x0 for v in variables]),
            residuals=np.array([]),
            cost=0.0,
            variables=variables, constraints=constraints,
        )

    col_for_var, n_cols = _link_group_index(variables)
    x0, lo, hi = _build_x0_and_bounds(variables, col_for_var, n_cols)

    # Track iteration count, last-known residuals, and the best (x, cost)
    # seen so far.  The best-tracking lets cancellation (StopIteration from
    # the callback) still leave the lattice in the best-of-run state rather
    # than whatever the last sampled x happened to be.
    state = {"iter": 0, "last_per": {}, "last_results": None,
             "best_x": None, "best_cost": float("inf"),
             # Sequential_scan sets this before each fn() call so the
             # callback can surface what's currently being scanned (which
             # element + which ADJUST attr).  Empty for other algorithms.
             "scan_info": {}}

    # Decide once: does the provided callback accept a 4th "info" arg
    # (richer per-eval payload with emittance, energy, current element)?
    # Old-style callbacks `cb(it, x, cost)` still work; new-style
    # `cb(it, x, cost, info)` gets the extras for live GUI display.
    _callback_accepts_info = False
    if callback is not None:
        try:
            import inspect as _inspect
            _sig = _inspect.signature(callback)
            _callback_accepts_info = len(_sig.parameters) >= 4
        except (TypeError, ValueError):
            _callback_accepts_info = False

    # Validate cost_solver early so a typo doesn't silently fall through
    # to the envelope path.
    if cost_solver not in ("envelope", "mp"):
        raise ValueError(
            f"unknown cost_solver {cost_solver!r}; "
            f"expected 'envelope' or 'mp'"
        )
    # None = "use each path's historical default": 1000 for the MP cost
    # solver, 1500 for the gradient+SC bunch.  An explicit value is
    # honoured by BOTH (review round 2: the gradient path used to
    # ignore it).
    _mp_default = mp_n_particles is None
    mp_n_particles = 1000 if _mp_default else int(mp_n_particles)
    if cost_solver == "mp" and int(mp_n_particles) <= 0:
        raise ValueError(
            f"cost_solver='mp' requires mp_n_particles >= 1, "
            f"got {mp_n_particles}"
        )
    # Parallel CMA-ES search scoring is an EXPLICIT choice (honesty
    # round, 2026-07-19): "auto" follows cost_solver faithfully — with
    # cost_solver="mp" the pool workers genuinely evaluate the
    # multiparticle objective; "envelope" is the fast envelope-guided
    # hybrid (envelope-scored search, mp-scored baseline/polish/final)
    # as a named opt-in.  The workers previously hardwired the envelope
    # cost regardless of cost_solver — a silent objective substitution.
    if cmaes_search_solver not in ("auto", "envelope", "mp"):
        raise ValueError(
            f"unknown cmaes_search_solver {cmaes_search_solver!r}; "
            f"expected 'auto', 'envelope' or 'mp'"
        )
    if cmaes_search_solver == "mp" and cost_solver != "mp":
        raise ValueError(
            "cmaes_search_solver='mp' with cost_solver='envelope': the "
            "search may not use a heavier model than the reported cost "
            "— set cost_solver='mp'"
        )
    _cmaes_search = (cost_solver if cmaes_search_solver == "auto"
                     else cmaes_search_solver)
    if (algorithm == "cmaes" and cmaes_search_solver != "auto"
            and cmaes_parallel != 0 and cmaes_parallel <= 1):
        # The search-solver override only exists inside the parallel
        # pool; the sequential branch always scores with cost_solver.
        # Every value that RESOLVES to one worker must trigger this:
        # n_workers = max(1, cmaes_parallel) for nonzero values, so
        # 1 AND all negatives run sequential (0 is the auto sentinel).
        # Documented in the help text, but say it at runtime too — a
        # user asking for the envelope-scored search without raising
        # the worker count would otherwise silently run the expensive
        # mp-scored one.
        import sys as _sys
        print(f"[match] cmaes_search_solver={cmaes_search_solver!r} "
              f"has NO EFFECT with cmaes_parallel={cmaes_parallel} "
              f"(resolves to sequential) — sequential CMA-ES scores "
              f"with cost_solver={cost_solver!r}; set cmaes_parallel>1 "
              f"(or 0 for auto worker count) to use the search "
              f"override", file=_sys.stderr)

    # Strict-by-default constraint audit (2026-07-11): a match must not
    # silently ignore a condition the user asked for.  Two failure modes
    # are caught here rather than warned about at evaluation time:
    #   1. stub constraints (parsed for round-trip, evaluator not
    #      implemented) carrying a nonzero weight;
    #   2. MIN_TRANSMISSION with the envelope cost solver, where
    #      apertures are no-ops and the residual is identically zero.
    # (Position constraints — DIAG_POSITION/SET_POSITION — used to be a
    # third mode; since the envelope tracks a first moment, 2026-07-19,
    # they fit under both cost solvers and are no longer audited.)
    # Pass allow_inert_constraints=True to restore the old warn-and-
    # continue behaviour (e.g. for legacy decks in the GUI).
    if not allow_inert_constraints:
        inert = []
        for c in constraints:
            is_stub = ("(stub)" in c.label
                       or "not implemented" in (c.notes or "")
                       or "not wired" in (c.notes or ""))
            if is_stub and float(c.weight) != 0.0:
                inert.append(f"{c.label} [stub evaluator]")
            # NOTE: position constraints (requires_mp — historical name)
            # are no longer envelope-inert: EnvelopeResults carries a
            # real first moment, so DIAG_POSITION / SET_POSITION fit
            # under BOTH cost solvers.  Only particle-loss observables
            # remain MP-only.
            elif (c.label.startswith("MIN_TRANSMISSION")
                    and cost_solver != "mp"):
                inert.append(f"{c.label} [inert: envelope mode tracks "
                             "no particle loss; use cost_solver='mp']")
            # SET_SIZE-family k2 != 0 asks for sizes INCLUDING the beam
            # centroid (TW manual); HELIX evaluates about the centroid
            # (the k2=0 convention) regardless — the flag is silently
            # remodelled.  Warned per collect since 2026-07; under the
            # audit it is a hard error unless explicitly allowed.
            elif (getattr(getattr(c, "source", None), "k2", 0)
                    and c.label.startswith("SET_SIZE")
                    and float(c.weight) != 0.0):
                inert.append(
                    f"{c.label} [k2={c.source.k2} not modelled: sizes "
                    "are evaluated about the beam centroid (k2=0 "
                    "convention); set k2=0 on the card]")
        if inert:
            raise ValueError(
                "match() would silently ignore active constraint(s): "
                + "; ".join(inert)
                + ".  Fix the lattice (set the card's weight to 0 or "
                "remove it), switch cost_solver, or pass "
                "allow_inert_constraints=True to proceed anyway."
            )

    # Envelope-scored pool workers track no particle loss —
    # MIN_TRANSMISSION would silently read zero in every worker.
    # Refuse that combination outright (independent of
    # allow_inert_constraints: it is a wrong answer, not a soft
    # degradation).  With mp-scored search the observable exists and
    # the constraint is honoured.
    if (algorithm == "cmaes" and cmaes_parallel != 1
            and _cmaes_search != "mp"
            and any(c.label.startswith("MIN_TRANSMISSION")
                    for c in constraints)):
        raise ValueError(
            "cmaes with cmaes_parallel != 1 and envelope-scored search "
            "cannot honour MIN_TRANSMISSION: the workers evaluate the "
            "envelope cost, which tracks no particle loss.  Use "
            "cost_solver='mp' (faithful parallel mp search), "
            "cmaes_parallel=1, or another algorithm."
        )

    def _residual_at(x):
        """Apply ``x``, run the forward pass, return
        ``(residual vector, per)``.

        Pure: no iteration counting and no callback — used for the final
        re-evaluation at the optimiser's solution.

        Dispatches on ``cost_solver``:
        * ``"envelope"`` (default): RMS sigma matrix via
          :class:`EnvelopeSolver` -- fast, linear SC kick, good for
          first-pass / linear lattices.
        * ``"mp"``: multi-particle PIC via
          :func:`linac_gen.cli.common.run_mp_sim` -- captures nonlinear
          SC, halo formation, and full 4-D phase-space evolution.
          Much slower (50-100x) but the cost reflects what MP-mode
          validation will report.

        Stashes the result in ``state["last_results"]`` for
        sequential_scan's reversal check.  Both forward passes return
        objects with the same ``emit_x``/``emit_y``/``emit_z``/
        ``emit_4d``/``ref_beta``/``ref_gamma``/``ref_w_kin`` fields the
        constraint evaluators read.
        """
        _apply_x(x, variables, col_for_var)
        if cost_solver == "mp":
            results = _run_mp(lattice, beam_cfg, space_charge=space_charge,
                              n_particles=mp_n_particles, seed=mp_seed,
                              sc_config=mp_sc_config,
                              step_config=mp_step_config)
        else:
            results = _run_envelope(lattice, beam_cfg,
                                    space_charge=space_charge)
        state["last_results"] = results
        return _evaluate_residuals(constraints, results, lattice)

    def fn(x):
        """Residual *vector* for ``least_squares`` — counts the evaluation,
        tracks best-so-far, and fires the progress callback."""
        residuals, per = _residual_at(x)
        state["iter"] += 1
        state["last_per"] = per
        cost_val = 0.5 * float(residuals @ residuals)
        if cost_val < state["best_cost"]:
            state["best_cost"] = cost_val
            state["best_x"] = np.asarray(x, dtype=float).copy()
        if callback is not None:
            # Build a per-step info dict for callbacks that want it
            # (live GUI displays).  Cheap: dict construction + a few
            # array index reads from state["last_results"] which was
            # just populated by _residual_at above.
            info_dict = None
            if _callback_accepts_info:
                lr = state.get("last_results")
                info_dict = {
                    "iter": state["iter"],
                    "cost": cost_val,
                    "best_cost": state["best_cost"],
                    **state.get("scan_info", {}),
                }
                if lr is not None:
                    # Full forward-sim result of THIS evaluation (a
                    # DiagnosticRecorder for cost_solver="mp", else
                    # EnvelopeResults) — consumers must treat it as
                    # READ-ONLY; every evaluation builds a fresh object,
                    # so a held reference is never mutated underneath.
                    # Powers the GUI's live match preview.
                    info_dict["results"] = lr
                if lr is not None:
                    try:
                        bg_out = (lr.ref_beta[-1] * lr.ref_gamma[-1]
                                  if lr.ref_beta and lr.ref_gamma else 1.0)
                        info_dict["emit_nx_out"] = (
                            float(lr.emit_x[-1]) * bg_out
                            if lr.emit_x else None)
                        info_dict["emit_ny_out"] = (
                            float(lr.emit_y[-1]) * bg_out
                            if lr.emit_y else None)
                        info_dict["emit_nz_out"] = (
                            float(lr.emit_z[-1]) * bg_out
                            if lr.emit_z else None)
                        info_dict["w_kin_out"] = (
                            float(lr.ref_w_kin[-1])
                            if lr.ref_w_kin else None)
                    except (AttributeError, IndexError, TypeError):
                        pass
            try:
                if _callback_accepts_info:
                    callback(state["iter"],
                             np.asarray(x, dtype=float).copy(),
                             cost_val, info_dict)
                else:
                    callback(state["iter"],
                             np.asarray(x, dtype=float).copy(),
                             cost_val)
            except StopIteration as exc:
                # GUI Stop button: callback raised StopIteration to abort
                # the match.  Before letting it propagate, restore the
                # lattice to the best-cost x we've seen so far -- so
                # downstream Apply / Save use the best result the matcher
                # found before cancellation.  Convert to _MatchCancelled
                # to bypass scipy.differential_evolution's batched
                # evaluator which silently swallows StopIteration
                # (PEP-479 / generator interop) and reports a misleading
                # "func must return scalar" RuntimeError.
                best_x = (state["best_x"] if state["best_x"] is not None
                          else np.asarray(x, dtype=float))
                _apply_x(np.asarray(best_x, dtype=float),
                         variables, col_for_var)
                raise _MatchCancelled(*exc.args) from exc
        return residuals

    def cost_fn(x):
        """Scalar objective for the global optimisers.  Wraps ``fn`` so the
        iteration counter and progress callback keep working unchanged."""
        residuals = fn(x)
        return 0.5 * float(residuals @ residuals)

    # Helper: when the user clicks Stop the callback raises StopIteration
    # which fn() catches, applies state["best_x"] to the lattice, and
    # re-raises.  Each algorithm branch wraps its main optimiser call so
    # the cancellation produces a synthetic ``res`` pointing at the
    # best-of-run x; the common tail then re-evaluates and returns a
    # proper MatchResult.  The GUI's _on_match_finished slot populates
    # tables + enables Apply normally -- "Stop" is now a soft-success,
    # not a failure.
    def _synthesize_cancelled_res():
        class _Res:
            pass
        r = _Res()
        r.x = np.asarray(
            state["best_x"] if state["best_x"] is not None else x0,
            dtype=float,
        )
        r.success = False
        r.message = (f"cancelled by user "
                     f"(best of {state['iter']} evals applied)")
        return r

    # ------------------------------------------------------------------
    # BASELINE PASS: evaluate the cost at x0 BEFORE any optimisation
    # step, using the chosen cost_solver.  This gives every algorithm:
    #
    # * A deterministic "starting cost" the user can compare against
    #   the final cost to verify the matcher actually improved (or
    #   correctly determined the lattice was already matched).
    # * A populated state["best_x"] = x0 + state["best_cost"] = cost(x0).
    #   The optimiser can only IMPROVE from here -- if it never finds
    #   a better x, the returned result is x0, no worse than the
    #   unmatched lattice.
    # * Consistency across algorithms: LS/DE/DA/gradient naturally
    #   evaluate at x0 as their first step, but CMA-ES samples
    #   N(x0, sigma) without hitting x0 exactly -- this baseline gives
    #   CMA-ES users a true x0 reference too.
    # * One extra forward pass per match (~30 s envelope, ~150 s MP).
    #   For matches running hundreds of evals this is rounding error.
    #
    # NOTE: sequential_scan does its own seed pass via fn(x0); we skip
    # the duplicate to avoid an extra MP eval (which could be expensive).
    # The seqscan branch reads state["last_results"] from this same call.
    baseline_cost: Optional[float] = None
    if algorithm != "sequential_scan":
        try:
            fn(x0)
            # state["best_cost"] now holds cost(x0); capture into a
            # locked-in variable so later improvements don't shadow it.
            baseline_cost = state["best_cost"]
            # SET_TWISS kaz/kbz baseline gate (2026-07): a degenerate
            # or absent longitudinal record at x0 means the z axes have
            # no gradient — refuse here (k2/stub-audit pattern) instead
            # of letting a warn-and-fill silently solve a different
            # problem than the deck requested.
            from linac_gen.matching.constraints import (
                audit_z_twiss_baseline)
            audit_z_twiss_baseline(constraints, state.get("last_results"),
                                   allow_inert_constraints)
        except _MatchCancelled:
            # User cancelled before the algorithm even started.  Best_x
            # is x0 (lattice already at x0 via fn() -> _residual_at ->
            # _apply_x).  Build the cancelled result directly and skip
            # the dispatch + common tail.
            class _BaselineRes:
                pass
            res = _BaselineRes()
            res.x = x0.copy()
            res.success = False
            res.message = "cancelled by user during baseline evaluation"
            x_final = np.asarray(res.x, dtype=float)
            residuals, per = _residual_at(x_final)
            cost = 0.5 * float(residuals @ residuals)
            baseline_cost_capture = state.get("best_cost", float("nan"))
            return MatchResult(
                success=False, message=res.message,
                n_iter=int(state["iter"]),
                elapsed_s=time.time() - t0, x0=x0,
                x_final=x_final, residuals=residuals, cost=cost,
                variables=variables, constraints=constraints,
                per_constraint_residuals=per,
                baseline_cost=(baseline_cost_capture
                               if math.isfinite(baseline_cost_capture)
                               else None),
            )

    if algorithm == "least_squares":
        try:
            res = least_squares(
                fn, x0, bounds=(lo, hi),
                method="trf", max_nfev=max_iter,
                xtol=xtol, ftol=ftol, gtol=1e-10,
            )
        except _MatchCancelled:
            res = _synthesize_cancelled_res()
    elif algorithm == "differential_evolution":
        _require_finite_bounds(algorithm, variables, col_for_var, lo, hi)
        try:
            res = differential_evolution(
                cost_fn, bounds=list(zip(lo, hi)),
                x0=x0, maxiter=max_iter, rng=0, polish=True,
            )
        except _MatchCancelled:
            res = _synthesize_cancelled_res()
    elif algorithm == "dual_annealing":
        _require_finite_bounds(algorithm, variables, col_for_var, lo, hi)
        try:
            res = dual_annealing(
                cost_fn, bounds=list(zip(lo, hi)),
                x0=x0, maxiter=max_iter, rng=0,
            )
        except _MatchCancelled:
            res = _synthesize_cancelled_res()
    elif algorithm == "cmaes":
        _require_finite_bounds(algorithm, variables, col_for_var, lo, hi)
        try:
            import cma
        except ImportError as exc:    # noqa: F841
            raise ImportError(
                "the 'cmaes' algorithm requires the 'cma' package; "
                "install with: pip install cma"
            )
        # Default popsize is `4 + floor(3·ln(N))` per Hansen.
        popsize = (cmaes_popsize if cmaes_popsize > 0
                   else max(4, 4 + int(math.floor(3 * math.log(max(n_cols, 1))))))
        # ``cmaes_sigma`` is documented as a FRACTION of the bound-box
        # width.  Pass per-column widths via CMA_stds so the effective
        # per-dimension step is sigma·width — previously the raw fraction
        # was used as an ABSOLUTE sigma in variable units, degenerating
        # CMA-ES to a local search on wide boxes (e.g. a [0, 50] T/m
        # gradient explored with σ=0.2 T/m) and mis-scaling mixed-unit
        # variable sets relative to each other.
        _widths = hi - lo
        _widths = np.where(np.isfinite(_widths) & (_widths > 0),
                           _widths, 1.0)
        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            float(cmaes_sigma),
            {
                "bounds": [lo.tolist(), hi.tolist()],
                "CMA_stds": _widths.tolist(),
                "maxiter": int(max_iter),
                "popsize": int(popsize),
                "tolx": float(xtol),
                "tolfun": float(ftol),
                "verbose": -9,
                "seed": 1,
            },
        )

        # Resolve parallel pool size.  Default (cmaes_parallel <= 1) is
        # sequential -- matches the previous behaviour exactly.
        # cmaes_parallel == 0 auto-detects to popsize capped at CPU
        # count - 1 (leave one core for the OS).
        import os as _os
        if cmaes_parallel == 0:
            n_workers = max(1, min(popsize,
                                   (_os.cpu_count() or 2) - 1))
            if n_workers == 1 and cmaes_search_solver != "auto":
                # Auto on a low-core host resolves to sequential — the
                # early <=1 notice can't see this, so say it here (the
                # same silent-sequential shape as the fixed -1 bug).
                import sys as _sys
                print(f"[match] cmaes_search_solver="
                      f"{cmaes_search_solver!r} has NO EFFECT: "
                      f"cmaes_parallel=0 auto-resolved to 1 worker on "
                      f"this host — sequential CMA-ES scores with "
                      f"cost_solver={cost_solver!r}", file=_sys.stderr)
        else:
            n_workers = max(1, int(cmaes_parallel))
        pool = None
        if n_workers > 1:
            try:
                from multiprocessing import Pool as _Pool
                pool = _Pool(
                    processes=n_workers,
                    initializer=_cmaes_pool_init,
                    initargs=(lattice, beam_cfg, variables, col_for_var,
                              space_charge, _cmaes_search,
                              dict(n_particles=mp_n_particles,
                                   seed=mp_seed,
                                   sc_config=mp_sc_config,
                                   step_config=mp_step_config)),
                )
            except Exception as exc:    # noqa: BLE001
                # Fall back to sequential (with the notice below) --
                # pickling can fail for lattices that carry unpicklable
                # extension state.  Sequential is always safe.
                import sys as _sys
                if cmaes_search_solver != "auto":
                    print(f"[match] cmaes_search_solver="
                          f"{cmaes_search_solver!r} has NO EFFECT: the "
                          f"worker pool could not be built — sequential "
                          f"CMA-ES scores with cost_solver="
                          f"{cost_solver!r}", file=_sys.stderr)
                print(f"[match] parallel CMA-ES unavailable "
                      f"({type(exc).__name__}: {exc}); "
                      f"falling back to sequential", file=_sys.stderr)
                pool = None
        if pool is not None and cost_solver == "mp":
            import sys as _sys
            if _cmaes_search == "mp":
                print(f"[match] parallel CMA-ES scoring the MULTI-"
                      f"PARTICLE cost in {n_workers} workers — for the "
                      f"fast envelope-guided hybrid pass "
                      f"cmaes_search_solver='envelope'",
                      file=_sys.stderr)
            else:
                print("[match] parallel CMA-ES hybrid: search scored "
                      "with the ENVELOPE cost (explicit opt-in); "
                      "baseline, polish and final residuals use mp",
                      file=_sys.stderr)

        cma_cancelled = False
        try:
            while not es.stop():
                xs = es.ask()
                if pool is not None:
                    # Pool returns results in order.  We update the
                    # state["iter"] counter and fire the progress
                    # callback once per candidate in the generation --
                    # using the worker-computed cost directly, so we
                    # don't pay the per-eval cost a second time.
                    fs = pool.map(_cmaes_pool_eval,
                                  [list(xx) for xx in xs])
                    # Cancellation check + best-x tracking.  In the
                    # parallel path workers compute fs without calling
                    # fn(), so state["best_x"] would never get updated
                    # the way it does in the sequential else-branch.
                    # We replicate fn()'s best-x bookkeeping here so a
                    # user-initiated Stop preserves the best CMA-ES
                    # sample across the generations evaluated so far.
                    for xx, ff in zip(xs, fs):
                        state["iter"] += 1
                        ff_val = float(ff)
                        if ff_val < state["best_cost"]:
                            state["best_cost"] = ff_val
                            state["best_x"] = np.asarray(
                                xx, dtype=float).copy()
                        if callback is not None:
                            try:
                                callback(state["iter"],
                                         np.asarray(xx, dtype=float).copy(),
                                         ff_val)
                            except StopIteration as exc:
                                # No fn() to restore the lattice for us;
                                # apply best_x ourselves before converting
                                # to _MatchCancelled.
                                if state["best_x"] is not None:
                                    _apply_x(state["best_x"],
                                             variables, col_for_var)
                                raise _MatchCancelled(*exc.args) from exc
                else:
                    fs = [cost_fn(np.asarray(xx, dtype=float)) for xx in xs]
                es.tell(xs, fs)
        except _MatchCancelled:
            # User cancelled inside the CMA-ES loop -- skip the refine
            # polish and synthesize the result from state["best_x"].
            cma_cancelled = True
        finally:
            # A worker that died mid-run can make pool.close() / .join()
            # raise (BrokenPipeError, OSError, etc.).  Swallow those --
            # the match is already done; an uncaught cleanup exception
            # would escape the QThread and terminate the GUI worker
            # without ever emitting finished/failed.
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                except Exception as exc:    # noqa: BLE001
                    import sys
                    print(f"[match] pool cleanup failed "
                          f"({type(exc).__name__}: {exc}); continuing",
                          file=sys.stderr)

        if cma_cancelled:
            res = _synthesize_cancelled_res()
        else:
            x_best = np.asarray(es.result.xbest, dtype=float)
            cma_msg = f"CMA-ES: {','.join(sorted(es.stop()))}"
            if pool is not None:
                cma_msg = f"{cma_msg}  (parallel x{n_workers})"

            # Optional Levenberg-Marquardt polish from the CMA-ES best.
            if refine:
                try:
                    res = least_squares(
                        fn, x_best, bounds=(lo, hi),
                        method="trf",
                        max_nfev=max(50, int(max_iter // 4)),
                        xtol=float(xtol), ftol=float(ftol), gtol=1e-10,
                    )
                    res_message = f"{cma_msg} + LS refine: {res.message}"
                    res_success = bool(res.success)
                    res_x = np.asarray(res.x, dtype=float)
                except _MatchCancelled:
                    # User cancelled DURING the LS polish.  state["best_x"]
                    # holds the lowest-cost x across BOTH the CMA-ES
                    # generation and any partial-polish samples (fn()
                    # updates it on every call), so use it unconditionally.
                    res_x = (state["best_x"] if state["best_x"] is not None
                             else x_best)
                    res_message = f"{cma_msg} + LS refine cancelled by user"
                    res_success = False
                except Exception as exc:    # noqa: BLE001
                    res_message = (f"{cma_msg} (LS refine failed: "
                                   f"{type(exc).__name__}: {exc})")
                    res_success = True
                    res_x = x_best
            else:
                res_message = cma_msg
                res_success = True
                res_x = x_best

            # Synthesize a duck-typed scipy-result for the common tail.
            class _Res:
                pass
            res = _Res()
            res.x = res_x
            res.success = res_success
            res.message = res_message
    elif algorithm == "bayesopt":
        # Gaussian-process Bayesian optimisation (BoTorch).  Reuses the
        # same cost_fn(x) seam as the other globals -- iteration counting,
        # callback, best-x tracking and StopIteration->_MatchCancelled all
        # happen inside cost_fn/fn.  Needs finite bounds like DE/DA/CMA-ES.
        _require_finite_bounds(algorithm, variables, col_for_var, lo, hi)
        from linac_gen.matching.bayesopt import run_bayesopt

        # Physics-informed warm start: when matching on the EXPENSIVE MP
        # cost, give BO a cheap ENVELOPE objective to pre-scout the box.
        # The prior eval is a side model -- it must NOT touch the iter
        # counter or the progress callback (those track the real,
        # expensive objective), so it bypasses fn()/cost_fn() and calls
        # the envelope forward pass directly.
        prior_cost_fn = None
        if bo_prior and cost_solver == "mp":
            def prior_cost_fn(x):  # noqa: F811 -- intentional local
                _apply_x(np.asarray(x, dtype=float), variables, col_for_var)
                results = _run_envelope(lattice, beam_cfg,
                                        space_charge=space_charge)
                resid, _per = _evaluate_residuals(constraints, results,
                                                  lattice)
                return 0.5 * float(resid @ resid)

        try:
            res = run_bayesopt(
                cost_fn, x0, lo, hi,
                max_iter=int(max_iter),
                n_init=int(bo_n_init),
                seed=0,
                prior_cost_fn=prior_cost_fn,
            )
            # Optional Levenberg-Marquardt polish from the BO best, mirroring
            # the CMA-ES branch -- BO localises the basin, LM nails the
            # bottom (cheap; skipped when refine=False).
            if refine:
                try:
                    # Clip the BO best into [lo, hi] before seeding the LS
                    # polish: BO legitimately lands *on* a one-sided MIN_*
                    # bound and float rounding can push the seed an epsilon
                    # outside, which makes least_squares reject the initial
                    # guess ("outside of provided bounds") and skip the
                    # polish entirely.  The clip is a no-op for interior
                    # optima.
                    bo_x = np.clip(np.asarray(res.x, dtype=float), lo, hi)
                    bo_msg = res.message
                    res = least_squares(
                        fn, bo_x, bounds=(lo, hi), method="trf",
                        max_nfev=max(50, int(max_iter // 4)),
                        xtol=float(xtol), ftol=float(ftol), gtol=1e-10,
                    )
                    res.message = f"{bo_msg} + LS refine: {res.message}"
                except _MatchCancelled:
                    res = _synthesize_cancelled_res()
                except Exception as exc:    # noqa: BLE001
                    # Refine failed -- keep the BO result (best_x already
                    # applied to the lattice by the common tail).
                    res.message = (f"{res.message} (LS refine failed: "
                                   f"{type(exc).__name__}: {exc})")
        except _MatchCancelled:
            res = _synthesize_cancelled_res()
    elif algorithm == "sequential_scan":
        # Physics-aware coordinate descent: walk elements one at a time,
        # bracket-scan each ADJUST parameter (or grouped pair), reverse
        # direction when both transverse and longitudinal normalised
        # emittance exceed the *input beam* emittance.  The user's
        # hand-coded recipe.
        from linac_gen.matching.variables import group_variables_by_element

        # The bracket-scan step is ``step_frac × (vmax - vmin)``, so
        # unbounded variables (vmin=-inf or vmax=inf) would produce
        # ``inf`` deltas and NaN-out the lattice on the very first
        # step.  Reject the configuration with a clear message --
        # users that hit this need to add finite bounds on their
        # ADJUST cards.
        _require_finite_bounds(algorithm, variables, col_for_var, lo, hi)

        n_steps = max(1, int(seqscan_steps))
        n_passes = max(1, int(seqscan_passes))
        step_frac = max(0.001, float(seqscan_step_frac))

        # 1. Seed reference run at x0 -- establishes the input emittance
        #    (arr[0]) used by the reversal criterion AND warms
        #    state["best_x"] = x0 + state["best_cost"] = cost(x0) so that:
        #    (a) the user sees the baseline cost as the first progress
        #        event, and (b) the optimiser can only IMPROVE: even if
        #        the scan never finds a better x, the returned result
        #        is x0 (no worse than unmatched).
        #    Uses fn() not _run_envelope -- fn() dispatches via
        #    cost_solver so MP-mode matching gets an MP seed, not a
        #    (cheap-but-wrong) envelope seed.
        #
        #    Cancellation handling: a user clicking Stop during the
        #    seed pass would raise _MatchCancelled out of fn().  Catch
        #    it here so cancellation flows through the same
        #    seq_cancelled path as cancellations during the scan loop
        #    -- otherwise the exception would escape match() entirely
        #    and the worker's table-populating success path would never
        #    fire.
        seq_cancelled = False
        try:
            fn(x0)
        except _MatchCancelled:
            seq_cancelled = True
        baseline_cost = (state["best_cost"]
                         if state["best_cost"] != float("inf")
                         else None)
        seed_res = state["last_results"]
        # Same SET_TWISS kaz/kbz baseline gate as the main path —
        # sequential_scan runs its own seed pass instead of the shared
        # baseline block.
        if not seq_cancelled:
            from linac_gen.matching.constraints import (
                audit_z_twiss_baseline)
            audit_z_twiss_baseline(constraints, seed_res,
                                   allow_inert_constraints)
        # Reversal threshold reference -- two physical interpretations:
        #   "input":     compare trial exit ε to BEAM INPUT ε (first
        #                envelope step).  Tight; for lattices that
        #                intrinsically grow ε, reversal fires constantly.
        #   "seed_exit": compare trial exit ε to the NOMINAL / unmatched
        #                lattice's exit ε (last envelope step at x0).
        #                Natural for emittance minimization with
        #                intrinsic growth -- reversal fires only when
        #                the trial is genuinely WORSE than baseline.
        if seqscan_threshold not in ("input", "seed_exit"):
            raise ValueError(
                f"seqscan_threshold must be 'input' or 'seed_exit', "
                f"got {seqscan_threshold!r}")
        if seed_res is None or seq_cancelled:
            # Cancellation arrived before the envelope/MP recorded any
            # output (worker requested interruption while fn() was
            # mid-eval, or fn() raised before _residual_at completed).
            # Skip the scan setup; the post-loop block synthesises
            # the cancelled result from state["best_x"].
            emit_x_thresh_norm = 0.0
            emit_z_thresh_norm = 0.0
        else:
            if seqscan_threshold == "seed_exit":
                # Sample-at-end normalisation: use the seed's exit
                # ref_beta / ref_gamma (the beam may have accelerated
                # through the lattice, so entry and exit βγ differ).
                seed_bg = (
                    seed_res.ref_beta[-1] * seed_res.ref_gamma[-1]
                    if seed_res.ref_beta and seed_res.ref_gamma else 1.0)
                emit_x_thresh_norm = (float(seed_res.emit_x[-1]) * seed_bg
                                      if seed_res.emit_x else 0.0)
                emit_z_thresh_norm = (float(seed_res.emit_z[-1]) * seed_bg
                                      if seed_res.emit_z else 0.0)
            else:  # "input"
                seed_bg_in = (
                    seed_res.ref_beta[0] * seed_res.ref_gamma[0]
                    if seed_res.ref_beta and seed_res.ref_gamma else 1.0)
                emit_x_thresh_norm = (float(seed_res.emit_x[0]) * seed_bg_in
                                      if seed_res.emit_x else 0.0)
                # emit_z is deg·MeV (clock-referenced): the entrance
                # threshold is compared against EXIT values, so express
                # it at the exit frequency — a FREQ jump inside the
                # line must not read as z-growth (mirrors the
                # MIN_EMIT_GROWTH Z anchor in constraints.py).
                from linac_gen.matching.constraints import z_clock_ratio
                emit_z_thresh_norm = (
                    float(seed_res.emit_z[0]) * seed_bg_in
                    * z_clock_ratio(seed_res)
                    if seed_res.emit_z else 0.0)

        # 2. Build element-ordered scan plan.  Filter by user selection.
        var_groups = group_variables_by_element(variables)
        selected_names = (set(seqscan_element_names)
                          if seqscan_element_names else None)

        # Pre-build variable -> column lookup (avoids O(n_vars²) scan
        # inside the inner loop via variables.index(var) calls).
        var_to_col = {id(v): col_for_var[i]
                      for i, v in enumerate(variables)}

        def _step_var_group(group_vars, direction, x_array):
            """Mutate x_array IN PLACE: step each variable in
            ``group_vars`` by direction × step_frac × bound_width."""
            for var in group_vars:
                col = var_to_col[id(var)]
                bound_width = max(1e-12, var.vmax - var.vmin)
                delta = direction * step_frac * bound_width
                x_array[col] = float(np.clip(
                    x_array[col] + delta, var.vmin, var.vmax,
                ))

        def _reversal_triggered(emit_x_out_norm, emit_z_out_norm):
            """True when the direction-reversal criterion fires.

            ``both_grew`` (default, matches the user's recipe):
                reverse only when *both* the transverse and longitudinal
                normalised emittance at end of line exceed their input
                values.  Pure exchange between planes (one up, one
                down) does NOT trigger reversal.
            ``any_grew``:
                stricter -- reverse on any plane growing above input.
            """
            if seqscan_reversal == "any_grew":
                return (emit_x_out_norm > emit_x_thresh_norm
                        or emit_z_out_norm > emit_z_thresh_norm)
            return (emit_x_out_norm > emit_x_thresh_norm
                    and emit_z_out_norm > emit_z_thresh_norm)

        # ``seq_cancelled`` was initialised earlier (before the seed
        # pass) -- if the user clicked Stop during the seed, it's
        # already True and we should skip the scan loop entirely.

        # Run the scan; if the callback raises StopIteration (user
        # clicked Stop) the exception propagates out of fn() through
        # all four for loops to this try/except, where we set the
        # flag and synthesize the cancelled result below.
        def _do_scan():
            nonlocal seq_cancelled
            if seq_cancelled:
                return
            try:
                for pass_idx in range(n_passes):
                    for elem, elem_vars in var_groups.items():
                        name = getattr(elem, "name", repr(elem))
                        if selected_names is not None and name not in selected_names:
                            continue
                        fmap_attrs = ("kb", "ke", "phase")
                        fmap_by_attr = {a: [v for v in elem_vars if v.attr == a]
                                        for a in fmap_attrs}
                        scan_groups = [fmap_by_attr[a] for a in fmap_attrs
                                       if fmap_by_attr[a]]
                        remaining = [v for v in elem_vars
                                     if v.attr not in fmap_attrs]
                        scan_groups.extend([v] for v in remaining)

                        for group in scan_groups:
                            if state["best_x"] is not None:
                                x_current = state["best_x"].copy()
                            else:
                                x_current = x0.copy()
                            direction = +1
                            group_attrs = [v.attr for v in group]
                            for step_idx in range(n_steps):
                                x_new = x_current.copy()
                                _step_var_group(group, direction, x_new)
                                # Publish the current scan context so the
                                # callback can show "FMAP_002 ke=0.31, pass
                                # 1/2, step 3/7" in the live GUI display.
                                state["scan_info"] = {
                                    "pass": pass_idx + 1,
                                    "total_passes": n_passes,
                                    "step": step_idx + 1,
                                    "total_steps": n_steps,
                                    "element_name": name,
                                    "attrs": group_attrs,
                                    "x_values": {
                                        v.attr: float(v.read())
                                        for v in group
                                    },
                                    "direction": int(direction),
                                }
                                # Snapshot best_x BEFORE fn() runs so a
                                # loss-rejected step can be rolled back
                                # without keeping the loss-induced ε
                                # improvement in state["best_x"].
                                pre_best_x = (state["best_x"].copy()
                                              if state["best_x"] is not None
                                              else None)
                                pre_best_cost = state["best_cost"]
                                fn(x_new)
                                last_r = state.get("last_results")
                                if last_r is not None:
                                    # Hard-rejection check: if the trial
                                    # caused beam loss, revert any best_x
                                    # update from this step and flip
                                    # direction.  The user's recipe: scan
                                    # only as far as no loss occurs.
                                    if seqscan_reject_loss:
                                        trans = getattr(last_r, "transmission",
                                                        None) or []
                                        if (trans
                                                and float(trans[-1])
                                                < float(seqscan_loss_threshold_pct)):
                                            # Roll back -- this step does
                                            # NOT get credit even if cost
                                            # dropped (because cost dropped
                                            # by killing particles).
                                            state["best_x"] = pre_best_x
                                            state["best_cost"] = pre_best_cost
                                            direction *= -1
                                            # Don't advance x_current --
                                            # next step retries from the
                                            # last loss-free point in the
                                            # opposite direction.
                                            continue
                                    bg_out = (
                                        last_r.ref_beta[-1] * last_r.ref_gamma[-1]
                                        if last_r.ref_beta and last_r.ref_gamma
                                        else 1.0)
                                    emit_x_out_norm = (
                                        float(last_r.emit_x[-1]) * bg_out
                                        if last_r.emit_x else 0.0)
                                    emit_z_out_norm = (
                                        float(last_r.emit_z[-1]) * bg_out
                                        if last_r.emit_z else 0.0)
                                    if _reversal_triggered(emit_x_out_norm,
                                                            emit_z_out_norm):
                                        direction *= -1
                                x_current = x_new
                        if state["best_x"] is not None:
                            _apply_x(state["best_x"], variables, col_for_var)
            except _MatchCancelled:
                # User clicked Stop mid-step.  The lattice was just
                # mutated to `x_new` by the in-flight fn() call; restore
                # it to the best-x found so far so the synthesized
                # MatchResult.x_final agrees with what the lattice
                # actually holds when control returns to the caller.
                seq_cancelled = True
                if state["best_x"] is not None:
                    _apply_x(state["best_x"], variables, col_for_var)

        _do_scan()

        if seq_cancelled:
            res = _synthesize_cancelled_res()
        else:
            # Synthesize a duck-typed scipy-result.  res.x = state["best_x"]
            # (the running best across all scans, or x0 if no
            # evaluations were made -- e.g. user selected zero elements).
            class _Res:
                pass
            res = _Res()
            res.x = (state["best_x"] if state["best_x"] is not None
                     else x0.copy())
            res.success = state["best_x"] is not None
            res.message = (f"sequential_scan: {n_passes} passes, "
                           f"{n_steps} steps/param, step_frac={step_frac}, "
                           f"reversal={seqscan_reversal}, "
                           f"threshold={seqscan_threshold}")
    else:  # gradient — exact-Jacobian Levenberg–Marquardt
        import torch
        from torch.autograd.functional import jacobian as _torch_jacobian
        from linac_gen.distributions.factory import create_beam
        from linac_gen.matching.torch_objective import (
            build_torch_residual, build_torch_residual_sc,
            check_gradient_supported,
        )
        # Reference particle, built exactly as the envelope forward pass.
        grad_ref = create_beam(beam_cfg, seed=42).ref
        check_gradient_supported(lattice, grad_ref, variables, constraints)
        if bool(getattr(beam_cfg, "continuous", False)):
            # The torch envelope model has no DC (continuous-beam)
            # space-charge branch — refuse rather than silently score
            # the match with bunched-ellipsoid physics.
            raise MatchingConfigError(
                "the 'gradient' algorithm does not support continuous "
                "(DC) beams — its torch envelope model only implements "
                "bunched space charge.  Use cost_solver='envelope' or "
                "'mp' with a different algorithm."
            )

        if space_charge:
            # Gradient matching THROUGH non-linear PIC space charge: the
            # residual tracks a macro-particle bunch with the differentiable
            # step tracker.  This is a genuinely different (non-linear) SC
            # model than the envelope forward pass, so the envelope
            # self-validation used by the no-SC path does not apply.
            # mp_sc_config / mp_n_particles are honoured when given
            # (review round 2 — they used to be silently ignored here);
            # the fallbacks are the historical CPU-friendly defaults.
            if mp_sc_config is not None:
                sc_cfg = mp_sc_config
            else:
                from linac_gen.core.config import SpaceChargeConfig
                sc_cfg = SpaceChargeConfig(
                    nx=32, ny=32, nz=32, grid_extent=4.0,
                    use_gpu="cpu", grid_mode="adaptive",
                )
            _bunch = 1500 if _mp_default else mp_n_particles
            r_torch = build_torch_residual_sc(
                lattice, beam_cfg, grad_ref, variables, constraints,
                col_for_var, n_cols, sc_cfg=sc_cfg, bunch_size=_bunch,
            )
        else:
            r_torch = build_torch_residual(
                lattice, beam_cfg, grad_ref, variables, constraints,
                col_for_var, n_cols,
            )
            # Self-validate: the torch residual must reproduce the numpy
            # matcher residual at x0, or the gradient match would be
            # solving a different problem than the other algorithms.
            r0_np, _ = _residual_at(x0)
            r0_t = r_torch(
                torch.as_tensor(x0, dtype=torch.float64)).detach().numpy()
            if r0_np.shape != r0_t.shape:
                raise ValueError(
                    "the 'gradient' algorithm could not reproduce this "
                    "lattice's matcher residual layout — use 'least_squares'"
                )
            mismatch = float(np.max(np.abs(r0_np - r0_t)))
            if mismatch > 1e-6 * (1.0 + float(np.max(np.abs(r0_np)))):
                raise ValueError(
                    f"the 'gradient' algorithm could not reproduce this "
                    f"lattice's matcher residuals (mismatch {mismatch:.2e}); "
                    f"use 'least_squares'"
                )

        def fn_grad(x):
            r = r_torch(
                torch.as_tensor(x, dtype=torch.float64)).detach().numpy()
            state["iter"] += 1
            cost_val = 0.5 * float(r @ r)
            # Track best-so-far (mirrors fn() above) so cancellation
            # via the callback restores the lattice to the best-x.
            if cost_val < state["best_cost"]:
                state["best_cost"] = cost_val
                state["best_x"] = np.asarray(x, dtype=float).copy()
            if callback is not None:
                try:
                    callback(state["iter"],
                             np.asarray(x, dtype=float).copy(),
                             cost_val)
                except StopIteration as exc:
                    # GUI Stop: restore the lattice to best-x before
                    # re-raising as _MatchCancelled (see fn() above
                    # for why we don't use bare StopIteration).
                    best_x = (state["best_x"] if state["best_x"] is not None
                              else np.asarray(x, dtype=float))
                    _apply_x(np.asarray(best_x, dtype=float),
                             variables, col_for_var)
                    raise _MatchCancelled(*exc.args) from exc
            return r

        def jac_grad(x):
            # vectorize=True batches the per-output vJPs; without it the SC
            # residual's autograd graph thrashes memory and can OOM.
            J = _torch_jacobian(
                r_torch, torch.as_tensor(x, dtype=torch.float64),
                vectorize=True)
            return np.asarray(J.detach().numpy(), dtype=float)

        try:
            res = least_squares(
                fn_grad, x0, jac=jac_grad, bounds=(lo, hi),
                method="trf", max_nfev=max_iter,
                xtol=xtol, ftol=ftol, gtol=1e-10,
            )
        except _MatchCancelled:
            res = _synthesize_cancelled_res()

        if space_charge:
            # Report the PIC residual at the solution — the shared
            # envelope re-evaluation below would report a different SC
            # model than the one the matcher actually optimised.
            x_final = np.asarray(res.x, dtype=float)
            _apply_x(x_final, variables, col_for_var)
            r_final = r_torch(
                torch.as_tensor(x_final, dtype=torch.float64)
            ).detach().numpy()
            cost = 0.5 * float(r_final @ r_final)
            return MatchResult(
                success=bool(res.success) or cost < 1e-10,
                message=str(res.message), n_iter=int(state["iter"]),
                elapsed_s=time.time() - t0, x0=x0, x_final=x_final,
                residuals=r_final, cost=cost,
                variables=variables, constraints=constraints,
                per_constraint_residuals={},
                baseline_cost=baseline_cost,
            )

    # Re-evaluate at the optimiser's solution: leaves the lattice in the
    # matched state and makes the reported residuals / per-constraint
    # breakdown correspond exactly to x_final for every algorithm.
    x_final = np.asarray(res.x, dtype=float)
    residuals, per = _residual_at(x_final)
    cost = 0.5 * float(residuals @ residuals)

    return MatchResult(
        success=bool(res.success) or cost < 1e-10,
        message=str(res.message),
        n_iter=int(state["iter"]),
        elapsed_s=time.time() - t0,
        x0=x0,
        x_final=x_final,
        residuals=residuals,
        cost=cost,
        baseline_cost=baseline_cost,
        variables=variables, constraints=constraints,
        per_constraint_residuals=per,
    )
