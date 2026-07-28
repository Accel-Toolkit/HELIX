"""``python -m linac_gen mo`` — multi-objective lattice design.

Explores the Pareto trade-off surface between two or more competing
objectives over the lattice's ADJUST decision variables, and writes the
non-dominated front to CSV (and optionally a PNG plot).

    python -m linac_gen mo lattice.dat \
        --objective emit_nz_growth --objective neg_exit_energy \
        --algorithm nsga2 --pop-size 24 --n-gen 15 \
        --energy 2.5 --current 5 --freq 162.5 --out pareto.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

from linac_gen.cli import common


def add_arguments(p) -> None:
    """Populate the ``mo`` sub-parser."""
    p.add_argument("input", nargs="?", default=None,
                   help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--objective", action="append", dest="objectives",
                   default=None, metavar="NAME",
                   help="objective to minimise (repeatable, >=2 required).  "
                        "See --list-objectives for the names.")
    p.add_argument("--list-objectives", action="store_true",
                   dest="list_objectives",
                   help="print the available objective names and exit")
    p.add_argument("--algorithm", choices=("nsga2", "qnehvi"),
                   default="nsga2",
                   help="nsga2 (genetic, default; cheap forward pass) or "
                        "qnehvi (Bayesian MO; sample-efficient for "
                        "expensive MP objectives)")
    p.add_argument("--pop-size", type=int, default=24, dest="pop_size",
                   help="NSGA-II population (qnehvi: initial Sobol design)")
    p.add_argument("--n-gen", type=int, default=15, dest="n_gen",
                   help="NSGA-II generations (qnehvi: BO iterations)")
    p.add_argument("--cost-solver", choices=("envelope", "mp"),
                   default="envelope", dest="cost_solver",
                   help="forward-pass model for objective evaluation")
    p.add_argument("--mp-n-particles", type=int, default=1000,
                   dest="mp_n_particles")
    p.add_argument("--space-charge", action="store_true", dest="space_charge",
                   help="DEPRECATED no-op: space charge is now automatic "
                        "whenever the beam current is > 0 (matching "
                        "'linac_gen run'); use --no-space-charge to opt "
                        "out")
    p.add_argument("--no-space-charge", action="store_true",
                   dest="no_space_charge",
                   help="disable space charge even when current > 0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pareto.csv",
                   help="CSV path for the Pareto front (default pareto.csv)")
    p.add_argument("--plot", default=None,
                   help="optional PNG path for a 2-D Pareto-front scatter "
                        "(first two objectives)")
    # beam overrides
    p.add_argument("--energy", type=float)
    p.add_argument("--current", type=float)
    p.add_argument("--freq", type=float)
    p.add_argument("--species")


def run(args) -> int:
    from linac_gen.matching.multiobjective import (
        pareto_optimize, OBJECTIVES, objective_labels,
    )

    if args.list_objectives:
        print("Available objectives (all minimised; smaller = better):")
        for name in sorted(OBJECTIVES):
            print(f"  {name:<20s} {OBJECTIVES[name][1]}")
        return 0

    if not args.input or not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.objectives or len(args.objectives) < 2:
        print("error: need >=2 --objective NAME (see --list-objectives)",
              file=sys.stderr)
        return 2

    try:
        lattice, beam_cfg, _conv = common.load_input(args.input)
        if args.energy is not None:
            beam_cfg.energy = args.energy
        if args.freq is not None:
            beam_cfg.frequency = args.freq
        if args.current is not None:
            beam_cfg.current = args.current
        if args.species is not None:
            beam_cfg.species = args.species
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Space charge is automatic from the physics (current > 0), matching
    # `linac_gen run` — the old opt-in --space-charge flag was the one
    # entry point where SC silently defaulted OFF for a current-carrying
    # beam.  --no-space-charge is the explicit opt-out.
    use_sc = float(beam_cfg.current or 0.0) > 0.0 \
        and not args.no_space_charge
    if args.space_charge:
        print("note: --space-charge is deprecated and a no-op — space "
              "charge is automatic when current > 0 (use "
              "--no-space-charge to disable)", file=sys.stderr)

    try:
        res = pareto_optimize(
            lattice, beam_cfg, args.objectives,
            algorithm=args.algorithm,
            space_charge=use_sc,
            cost_solver=args.cost_solver,
            mp_n_particles=args.mp_n_particles,
            pop_size=args.pop_size, n_gen=args.n_gen, seed=args.seed,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    labels = objective_labels(res.objective_names)
    print(f"[mo] {res.message}")
    print(f"[mo] {res.n_eval} evaluations -> {len(res.pareto_F)} "
          f"Pareto-optimal designs")
    print(f"[mo] objectives: {', '.join(res.objective_names)}")

    # Write the front to CSV: objective columns then decision columns.
    # column_variable_labels() is link-group-deduplicated so the header
    # width matches the pareto_x columns even when ADJUST cards are linked.
    import csv
    var_labels = res.column_variable_labels()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(res.objective_names) + var_labels)
        for f_row, x_row in zip(res.pareto_F, res.pareto_x):
            w.writerow([f"{v:.6g}" for v in f_row]
                       + [f"{v:.6g}" for v in x_row])
    print(f"[mo] wrote {args.out}")

    if args.plot and res.pareto_F.shape[1] >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(res.all_F[:, 0], res.all_F[:, 1], s=10,
                       c="#bbbbbb", label="all evaluated")
            ax.scatter(res.pareto_F[:, 0], res.pareto_F[:, 1], s=40,
                       c="#f97316", label="Pareto front")
            ax.set_xlabel(f"{res.objective_names[0]}  ({labels[0]})")
            ax.set_ylabel(f"{res.objective_names[1]}  ({labels[1]})")
            ax.legend()
            ax.set_title("Pareto front")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"[mo] wrote {args.plot}")
        except Exception as exc:  # noqa: BLE001
            print(f"[mo] plot skipped ({type(exc).__name__}: {exc})")

    return 0
