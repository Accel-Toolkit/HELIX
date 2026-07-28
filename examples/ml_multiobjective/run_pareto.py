#!/usr/bin/env python
"""Demo: multi-objective lattice design -- the Pareto trade-off between
longitudinal emittance growth and exit energy.

These two objectives genuinely *compete*: pushing the cavities for more
exit energy means running further off-crest, which grows the longitudinal
emittance.  There is no single best design -- there is a *front* of
non-dominated compromises, and the designer picks a knee point.

Runs NSGA-II (genetic, cheap envelope forward pass) and prints the front;
optionally also runs qNEHVI (Bayesian multi-objective) to show how few
evaluations it needs.  Saves the front to CSV and a PNG scatter.

    python examples/ml_multiobjective/run_pareto.py

Equivalent CLI:

    python -m linac_gen mo examples/ml_multiobjective/mo_demo.dat \
        --objective emit_nz_growth --objective neg_exit_energy \
        --algorithm nsga2 --pop-size 24 --n-gen 15 \
        --energy 2.5 --current 5 --freq 162.5 \
        --out pareto.csv --plot pareto.png
"""
from __future__ import annotations

import copy
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching.multiobjective import pareto_optimize, objective_labels

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = os.path.join(HERE, "mo_demo.dat")
OBJECTIVES = ["emit_nz_growth", "neg_exit_energy"]

BEAM = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=1000, distribution="waterbag",
    emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
    emit_ny=0.30, alpha_y=+2.0, beta_y=0.05,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)


def main():
    print(f"Lattice: {LATTICE}")
    labels = objective_labels(OBJECTIVES)
    print(f"Objectives (both minimised):")
    for n, lab in zip(OBJECTIVES, labels):
        print(f"  {n:<18s} {lab}")
    print()

    lat, _ = parse_tracewin(LATTICE)
    res = pareto_optimize(
        copy.deepcopy(lat), copy.deepcopy(BEAM), OBJECTIVES,
        algorithm="nsga2", pop_size=24, n_gen=15, seed=0,
    )
    print(f"NSGA-II: {res.n_eval} evals -> {len(res.pareto_F)} "
          f"Pareto-optimal designs\n")

    print(f"{'emit_nz_growth':>16}{'exit_energy[MeV]':>18}")
    print("-" * 34)
    for f in res.pareto_F:
        # neg_exit_energy is stored negated -> flip the sign for display.
        print(f"{f[0]:>16.4f}{-f[1]:>18.4f}")

    # Knee point: closest (after min-max normalising both objectives) to
    # the ideal corner -- a reasonable "balanced" pick off the front.
    F = res.pareto_F.copy()
    rng = F.max(axis=0) - F.min(axis=0)
    rng[rng == 0] = 1.0
    norm = (F - F.min(axis=0)) / rng
    knee = int(((norm ** 2).sum(axis=1) ** 0.5).argmin())
    print(f"\nKnee-point design: emit_nz_growth={res.pareto_F[knee,0]:.4f}, "
          f"exit_energy={-res.pareto_F[knee,1]:.4f} MeV")
    # column_variables() is link-group-deduplicated so it lines up with the
    # pareto_x columns even when ADJUST cards are ganged (it's just the full
    # list when nothing is linked).
    for v, x in zip(res.column_variables(), res.pareto_x[knee]):
        print(f"    {v.label:<22s} = {x:.4f}")

    # Save CSV + PNG (best effort).
    import csv
    out_csv = os.path.join(HERE, "pareto.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(OBJECTIVES + res.column_variable_labels())
        for fr, xr in zip(res.pareto_F, res.pareto_x):
            w.writerow([f"{v:.6g}" for v in fr] + [f"{v:.6g}" for v in xr])
    print(f"\nWrote {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(res.all_F[:, 0], -res.all_F[:, 1], s=10, c="#bbbbbb",
                   label="all evaluated")
        ax.scatter(res.pareto_F[:, 0], -res.pareto_F[:, 1], s=45,
                   c="#f97316", label="Pareto front")
        ax.scatter([res.pareto_F[knee, 0]], [-res.pareto_F[knee, 1]],
                   s=90, marker="*", c="#22d3ee", label="knee point")
        ax.set_xlabel("longitudinal emittance growth  (out / in)")
        ax.set_ylabel("exit kinetic energy  [MeV]")
        ax.set_title("Pareto front: ε_z growth vs exit energy")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(HERE, "pareto.png")
        fig.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {type(exc).__name__}: {exc})")

    print("\nTakeaway: there is no single 'best' design -- the front shows\n"
          "the achievable trade-off, and the knee point is a balanced pick.\n"
          "Swap --algorithm qnehvi (Bayesian MO) to reach a comparable\n"
          "front in far fewer evaluations when the forward pass is "
          "expensive.")


if __name__ == "__main__":
    main()
