#!/usr/bin/env python
"""Demo: physics-informed warm start for the Bayesian-optimisation matcher
on an EXPENSIVE (multi-particle) objective.

When ``cost_solver="mp"`` each forward pass is a full PIC simulation
(seconds-minutes on a real lattice).  ``bo_prior=True`` first scouts the
parameter box with the *cheap* envelope cost and seeds the GP's initial
design with the envelope-good points -- so the expensive MP objective is
queried where it is most likely to pay off.

This compares ``bo_prior=False`` vs ``True`` on a small MP match and
reports evals / final cost / wall-time.  Settings are deliberately tiny
so it finishes in a couple of minutes; the *gap* widens as the MP cost
grows relative to the envelope cost (i.e. on real high-current lattices).

    python examples/ml_bayesopt/run_bayesopt_prior.py

Equivalent CLI:

    python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
        --algorithm bayesopt --cost-solver mp --mp-n-particles 200 \
        --max-iter 6 --bo-prior \
        --energy 2.5 --current 5 --frequency 162.5 --space-charge --report
"""
from __future__ import annotations

import copy
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = os.path.join(HERE, "bo_demo.dat")

BEAM = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=200, distribution="waterbag",
    emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
    emit_ny=0.30, alpha_y=+2.0, beta_y=0.05,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)

# Tiny budget so the demo finishes quickly.  Raise for a real study.
MAX_ITER = 6
MP_N = 200


def _run(bo_prior):
    lat, _ = parse_tracewin(LATTICE)
    cfg = copy.deepcopy(BEAM)
    t0 = time.time()
    r = match(
        lat, cfg, space_charge=True,
        algorithm="bayesopt", max_iter=MAX_ITER, refine=False,
        cost_solver="mp", mp_n_particles=MP_N, mp_seed=42,
        bo_prior=bo_prior,
    )
    return {
        "prior": bo_prior, "evals": r.n_iter,
        "baseline": r.baseline_cost, "final": r.cost,
        "elapsed": time.time() - t0,
    }


def main():
    print(f"Lattice: {LATTICE}")
    print(f"MP cost solver, {MP_N} particles, {MAX_ITER} BO iters, "
          f"space charge ON.\nComparing physics-informed warm start "
          f"off vs on...\n")

    runs = [_run(False), _run(True)]

    print(f"{'bo_prior':<12}{'evals':>8}{'baseline':>14}"
          f"{'final':>14}{'sec':>9}")
    print("-" * 57)
    for r in runs:
        base = "n/a" if r["baseline"] is None else f"{r['baseline']:.3e}"
        print(f"{str(r['prior']):<12}{r['evals']:>8}{base:>14}"
              f"{r['final']:>14.3e}{r['elapsed']:>9.1f}")

    print("\nThe warm start spends a few cheap envelope evals up front to\n"
          "place the expensive MP samples better.  On this tiny lattice the\n"
          "envelope and MP costs are close, so the benefit is modest; it\n"
          "grows with the MP/envelope cost ratio (high-current, many-element\n"
          "lattices where MP is 50-100x the envelope cost).")


if __name__ == "__main__":
    main()
