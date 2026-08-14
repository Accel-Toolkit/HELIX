"""``python -m linac_gen.matching`` — drive the matcher from the shell.

Usage
-----
::

    python -m linac_gen.matching <input.dat> [--out <out.dat>]
                                  [--space-charge] [--current 5.0]
                                  [--algorithm least_squares]
                                  [--max-iter 200]
                                  [--report] [--no-write]

The matcher reads ``ADJUST_*`` and ``SET_*`` commands from the lattice,
runs the selected optimiser (``--algorithm``) against an envelope forward
simulation, writes the matched lattice back to ``--out`` (or
``<input>.matched.dat`` by default), and optionally prints a residual
report.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from linac_gen.core.config import BeamConfig
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin
from linac_gen.matching import MATCH_ALGORITHMS, match


def _build_default_beam_cfg(args, freq_hz: float | None) -> BeamConfig:
    """Return a sensible default ``BeamConfig`` when the user has not
    supplied one — the matcher needs *something* to seed the envelope."""
    return BeamConfig(
        species=args.species, energy=args.energy,
        frequency=(freq_hz if freq_hz else args.frequency),
        current=args.current, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=args.emit_x, alpha_x=args.alpha_x, beta_x=args.beta_x,
        emit_ny=args.emit_y, alpha_y=args.alpha_y, beta_y=args.beta_y,
        emit_z=args.emit_z, alpha_z=args.alpha_z, beta_z=args.beta_z,
    )


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m linac_gen.matching",
        description="Run Linac_Gen's matcher on a TraceWin .dat lattice.",
    )
    p.add_argument("input", type=str, help="path to lattice .dat")
    p.add_argument("--out", type=str, default=None,
                   help="output path (default: <input>.matched.dat)")
    p.add_argument("--space-charge", action="store_true",
                   help="enable SC during forward passes")
    p.add_argument("--diag-targets", type=str, default=None, metavar="PATH",
                   help="external BPM-target file (x_mm y_mm [weight] per "
                        "is_bpm marker, lattice order; nan = plane free); "
                        "overrides DIAG_POSITION operands.  Position "
                        "constraints need --cost-solver mp.")
    p.add_argument("--current", type=float, default=0.0,
                   help="beam peak current in mA (default 0)")
    p.add_argument("--algorithm", default="least_squares",
                   choices=MATCH_ALGORITHMS,
                   help="optimiser: least_squares (default, local), "
                        "differential_evolution / dual_annealing / cmaes / "
                        "bayesopt (global, need finite ADJUST bounds), or "
                        "gradient (local, exact-Jacobian).  bayesopt = "
                        "Gaussian-process Bayesian optimisation, best for "
                        "expensive (MP) + multimodal matches.")
    p.add_argument("--max-iter", type=int, default=200,
                   help="least_squares: max function evaluations; global "
                        "algorithms: max generations / annealing iterations")
    p.add_argument("--xtol", type=float, default=1e-8)
    p.add_argument("--ftol", type=float, default=1e-8)
    p.add_argument("--cmaes-sigma", dest="cmaes_sigma", type=float,
                   default=0.2,
                   help="CMA-ES initial step (fraction of bound width)")
    p.add_argument("--cmaes-popsize", dest="cmaes_popsize", type=int,
                   default=0,
                   help="CMA-ES population size; 0 = package default "
                        "(4 + floor(3·ln(N)))")
    p.add_argument(
        "--cmaes-parallel", dest="cmaes_parallel", type=int, default=1,
        help="Evaluate the CMA-ES population in parallel across N "
             "worker processes.  Default 1 = sequential (matches the "
             "non-parallel result exactly).  0 = auto-detect "
             "(min(popsize, cpu_count-1)).  Only worth >1 when the "
             "per-eval cost is large (RK4-driven FieldMap3D lattices); "
             "for surrogate-active matches the fork overhead can "
             "exceed the speedup.")
    p.add_argument(
        "--cmaes-search-solver", dest="cmaes_search_solver",
        choices=("auto", "envelope", "mp"), default="auto",
        help="Objective the parallel CMA-ES workers score the search "
             "with.  'auto' (default) follows --cost-solver faithfully "
             "— parallel + mp genuinely evaluates the multiparticle "
             "objective.  'envelope' is the fast envelope-guided "
             "hybrid (search scored by the envelope; baseline/polish/"
             "final residuals still use the requested cost solver) as "
             "an explicit opt-in.  No effect with --cmaes-parallel 1.")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="skip the LS polish after CMA-ES / bayesopt")
    p.set_defaults(refine=True)
    # Bayesian-optimisation options (engine.py: bo_* kwargs).
    p.add_argument("--bo-prior", dest="bo_prior", action="store_true",
                   help="bayesopt: physics-informed warm start -- use the "
                        "cheap envelope cost to pre-scout the box before "
                        "the expensive MP objective.  Only active with "
                        "--algorithm bayesopt AND --cost-solver mp.")
    p.add_argument("--bo-n-init", dest="bo_n_init", type=int, default=0,
                   help="bayesopt: number of initial space-filling (Sobol) "
                        "samples before the GP is first fit.  0 = default "
                        "min(2*N+2, 12).")
    # Cost-solver / MP forward-pass options (matches the GUI's
    # Cost-solver group; engine.py: cost_solver / mp_* kwargs).
    p.add_argument("--cost-solver", dest="cost_solver",
                   choices=("envelope", "mp"), default="envelope",
                   help="forward-pass model used to compute the cost: "
                        "'envelope' (fast linear) or 'mp' (full PIC, "
                        "physics-accurate but 50-100x slower per eval)")
    p.add_argument("--mp-n-particles", dest="mp_n_particles", type=int,
                   default=None,
                   help="macroparticle count for particle-based costs "
                        "(default: 1000 for the MP cost solver, 1500 "
                        "for the gradient+SC bunch; lower = faster but "
                        "noisier)")
    p.add_argument("--mp-seed", dest="mp_seed", type=int, default=42,
                   help="MP cost-solver: fixed seed for reproducible "
                        "per-evaluation cost (default 42)")
    p.add_argument("--mp-grid", dest="mp_grid", type=int, default=None,
                   help="PIC grid size per axis (nx=ny=nz) for "
                        "particle-based costs — the MP cost solver and "
                        "the gradient+SC bunch (default: the engine's "
                        "32^3 fallback)")
    p.add_argument("--mp-extent", dest="mp_extent", type=float,
                   default=None,
                   help="PIC grid half-extent in beam sigmas for "
                        "particle-based costs (default: the engine's "
                        "±4σ fallback)")
    # Sequential-scan algorithm options (engine.py: seqscan_* kwargs).
    p.add_argument("--seqscan-passes", dest="seqscan_passes", type=int,
                   default=2,
                   help="sequential_scan: full passes through the element "
                        "list (default 2)")
    p.add_argument("--seqscan-steps", dest="seqscan_steps", type=int,
                   default=7,
                   help="sequential_scan: bracket steps per parameter "
                        "per pass (default 7)")
    p.add_argument("--seqscan-step-frac", dest="seqscan_step_frac",
                   type=float, default=0.10,
                   help="sequential_scan: per-step displacement as "
                        "fraction of (vmax-vmin) (default 0.10)")
    p.add_argument("--seqscan-reversal", dest="seqscan_reversal",
                   choices=("both_grew", "any_grew"), default="both_grew",
                   help="sequential_scan: when to flip direction.  "
                        "'both_grew' = both transverse AND longitudinal ε "
                        "above reference (allows plane-exchange).  "
                        "'any_grew' = either above reference (stricter).")
    p.add_argument("--seqscan-threshold", dest="seqscan_threshold",
                   choices=("input", "seed_exit"), default="input",
                   help="sequential_scan: reference for the reversal "
                        "criterion.  'input' = compare to beam input ε "
                        "(tight; default).  'seed_exit' = compare to "
                        "nominal/unmatched lattice's exit ε (natural for "
                        "ε minimization with intrinsic growth).")
    p.add_argument("--seqscan-element", dest="seqscan_element_names",
                   action="append", default=None, metavar="NAME",
                   help="sequential_scan: only scan elements with these "
                        "names (repeatable).  Omit to scan all elements "
                        "carrying ADJUST cards.")
    p.add_argument("--seqscan-reject-loss", dest="seqscan_reject_loss",
                   action="store_true",
                   help="sequential_scan: HARD-reject any step whose "
                        "trial causes beam loss (transmission < "
                        "--seqscan-loss-threshold-pct).  The step is "
                        "rolled back -- best_x is NOT updated to it "
                        "even if cost dropped -- and direction is "
                        "flipped.  Closes the gameability where a low-"
                        "ε halo-clip would otherwise be preferred.  "
                        "Requires --cost-solver=mp (env mode doesn't "
                        "track losses; flag is silently inert there).")
    p.add_argument("--seqscan-loss-threshold-pct",
                   dest="seqscan_loss_threshold_pct",
                   type=float, default=100.0,
                   help="sequential_scan: transmission floor for the "
                        "loss rejection check (default 100.0 = any drop "
                        "below 100%% triggers rejection).  Loosen to "
                        "e.g. 99.9 to tolerate sub-permille losses.")
    p.add_argument("--report", action="store_true",
                   help="print full variable / constraint report after run")
    p.add_argument("--no-write", action="store_true",
                   help="skip writing the matched .dat (just report)")
    p.add_argument("--allow-inert-constraints",
                   dest="allow_inert_constraints", action="store_true",
                   help="proceed even when an active constraint would be "
                        "silently ignored (a stub-evaluator card carrying a "
                        "nonzero weight, or MIN_TRANSMISSION under the "
                        "envelope cost-solver).  Default: refuse with an "
                        "explanatory error.  Mirrors the match() kwarg of "
                        "the same name.")
    # Beam defaults — minimal scaffolding when the lattice does not carry
    # a beam-config of its own.
    p.add_argument("--species", default="proton")
    p.add_argument("--energy", type=float, default=3.0,
                   help="kinetic energy MeV")
    p.add_argument("--frequency", type=float, default=352.21,
                   help="reference RF frequency MHz")
    p.add_argument("--alpha-x", dest="alpha_x", type=float, default=0.0)
    p.add_argument("--beta-x",  dest="beta_x",  type=float, default=1.0)
    p.add_argument("--emit-x",  dest="emit_x",  type=float, default=0.25)
    p.add_argument("--alpha-y", dest="alpha_y", type=float, default=0.0)
    p.add_argument("--beta-y",  dest="beta_y",  type=float, default=1.0)
    p.add_argument("--emit-y",  dest="emit_y",  type=float, default=0.25)
    p.add_argument("--alpha-z", dest="alpha_z", type=float, default=0.0)
    p.add_argument("--beta-z",  dest="beta_z",  type=float, default=1.0)
    p.add_argument("--emit-z",  dest="emit_z",  type=float, default=0.3)
    return p


def main(argv: list[str] | None = None) -> int:
    from linac_gen.cli.common import make_console_encoding_safe
    make_console_encoding_safe()
    parser = _make_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    lattice, meta = parse_tracewin(args.input)
    if meta.get("warnings"):
        print(f"[parse] {len(meta['warnings'])} warnings; first: "
              f"{meta['warnings'][0]}", file=sys.stderr)

    if args.diag_targets:
        from linac_gen.io.diag_targets import (apply_diag_targets,
                                               load_diag_targets)
        try:
            n = apply_diag_targets(lattice,
                                   load_diag_targets(args.diag_targets))
        except ValueError as exc:
            print(f"error: --diag-targets: {exc}", file=sys.stderr)
            return 2
        print(f"[targets] applied {n} BPM target row(s) from "
              f"{args.diag_targets}")

    beam_cfg = _build_default_beam_cfg(args, freq_hz=None)

    print(f"[match] {len(lattice.elements)} elements; "
          f"algorithm={args.algorithm}; "
          f"cost_solver={args.cost_solver}; "
          f"running {'with' if args.space_charge else 'without'} SC")

    # PIC mesh override for particle-based costs (MP cost solver and the
    # gradient+SC bunch).  None -> the engine's documented fallbacks.
    mp_sc_config = None
    if args.mp_grid is not None or args.mp_extent is not None:
        from linac_gen.core.config import SpaceChargeConfig
        _nx = int(args.mp_grid) if args.mp_grid is not None else 32
        mp_sc_config = SpaceChargeConfig(
            nx=_nx, ny=_nx, nz=_nx,
            grid_extent=(float(args.mp_extent)
                         if args.mp_extent is not None else 4.0),
            use_gpu="cpu", grid_mode="adaptive",
        )

    result = match(lattice, beam_cfg,
                   space_charge=args.space_charge,
                   algorithm=args.algorithm,
                   max_iter=args.max_iter,
                   xtol=args.xtol, ftol=args.ftol,
                   cmaes_sigma=args.cmaes_sigma,
                   cmaes_popsize=args.cmaes_popsize,
                   cmaes_parallel=args.cmaes_parallel,
                   cmaes_search_solver=args.cmaes_search_solver,
                   refine=args.refine,
                   bo_prior=args.bo_prior,
                   bo_n_init=args.bo_n_init,
                   cost_solver=args.cost_solver,
                   mp_n_particles=args.mp_n_particles,
                   mp_seed=args.mp_seed,
                   mp_sc_config=mp_sc_config,
                   seqscan_passes=args.seqscan_passes,
                   seqscan_steps=args.seqscan_steps,
                   seqscan_step_frac=args.seqscan_step_frac,
                   seqscan_reversal=args.seqscan_reversal,
                   seqscan_threshold=args.seqscan_threshold,
                   seqscan_element_names=args.seqscan_element_names,
                   seqscan_reject_loss=args.seqscan_reject_loss,
                   seqscan_loss_threshold_pct=args.seqscan_loss_threshold_pct,
                   allow_inert_constraints=args.allow_inert_constraints)

    if result.baseline_cost is not None:
        print(f"[match] baseline cost (at x0) = {result.baseline_cost:.4e}")

    print(f"[match] {'OK' if result.success else 'FAILED'} -- "
          f"{result.n_iter} iters, cost {result.cost:.4e}, "
          f"{result.elapsed_s:.2f}s")
    if not result.success and not result.variables:
        # Engine returned a "no work to do" result; surface and continue
        # so the writer still emits a clean .dat with original values.
        print(f"[match] {result.message}")

    if args.report:
        print()
        print(result.report())

    if not args.no_write:
        out_path = args.out or str(Path(args.input).with_suffix(".matched.dat"))
        write_tracewin(lattice, out_path)
        print(f"[match] wrote {out_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
