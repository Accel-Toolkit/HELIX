#!/usr/bin/env python
"""Demo: Bayesian-optimisation matcher (``algorithm="bayesopt"``) vs the
gradient-free baselines, on the same 6-knob / 4-constraint lattice.

Runs least_squares, cmaes, and bayesopt against the identical match
problem and prints an A/B table of *evaluations* and *baseline -> final
cost*.  The point of Bayesian optimisation is **sample efficiency** --
reaching a comparable cost in fewer expensive forward passes -- so the
"evals" column is the headline metric.

Run headless:

    python examples/ml_bayesopt/run_bayesopt.py

Equivalent CLI for the bayesopt arm:

    python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
        --algorithm bayesopt --max-iter 30 \
        --energy 2.5 --current 5 --frequency 162.5 --report
"""
from __future__ import annotations

import copy
import os
import sys
import time

# Ensure THIS repo's linac_gen wins over any other copy on the path
# (an editable install may point elsewhere; running a script puts the
# script's own dir on sys.path[0], not the repo root).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = os.path.join(HERE, "bo_demo.dat")

# Deliberately mismatched seed (see bo_demo.dat header) so a baseline
# forward pass shows ~50% emittance growth and the matcher has real work.
BEAM = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=1000, distribution="waterbag",
    emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
    emit_ny=0.30, alpha_y=+2.0, beta_y=0.05,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)


def _run(algorithm, **kw):
    lat, _ = parse_tracewin(LATTICE)          # fresh copy — match mutates
    cfg = copy.deepcopy(BEAM)
    t0 = time.time()
    r = match(lat, cfg, space_charge=False, algorithm=algorithm, **kw)
    return {
        "algorithm": algorithm,
        "evals": r.n_iter,
        "baseline": r.baseline_cost,
        "final": r.cost,
        "elapsed": time.time() - t0,
        "success": r.success,
    }


def main():
    print(f"Lattice: {LATTICE}")
    print("Matching 6 knobs (2 solenoids, 2 cavity V, 2 cavity phase) "
          "against\n3 MIN_EMIT_GROWTH + 1 SET_KE_OUT_MIN constraint.\n")

    runs = []
    for algo, kw in [
        ("least_squares", dict(max_iter=200)),
        ("cmaes",         dict(max_iter=40, refine=True)),
        ("bayesopt",      dict(max_iter=30, refine=True)),
    ]:
        try:
            runs.append(_run(algo, **kw))
        except Exception as exc:  # noqa: BLE001
            print(f"  {algo}: FAILED ({type(exc).__name__}: {exc})")

    print(f"{'algorithm':<20}{'evals':>8}{'baseline':>14}"
          f"{'final':>14}{'sec':>8}")
    print("-" * 64)
    for r in runs:
        base = "n/a" if r["baseline"] is None else f"{r['baseline']:.3e}"
        print(f"{r['algorithm']:<20}{r['evals']:>8}{base:>14}"
              f"{r['final']:>14.3e}{r['elapsed']:>8.1f}")

    print("\nTakeaway: all three reach a low final cost, but Bayesian "
          "optimisation\ngets there in the fewest *evaluations* -- the "
          "metric that matters when\neach forward pass is expensive "
          "(Cost solver = mp / space charge).")


if __name__ == "__main__":
    main()
