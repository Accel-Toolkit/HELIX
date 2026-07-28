#!/usr/bin/env python
"""Demo: the MIN_TRANSMISSION soft-loss matching constraint.

Pure emittance minimisation has a notorious failure mode: the matcher can
LOWER the emittance by letting a tight aperture scrape off the beam halo
-- a smaller epsilon bought with lost beam.  `MIN_TRANSMISSION threshold
weight` adds a one-sided penalty that fires only when transmission drops
below the threshold, forbidding that trade.  It is MULTI-PARTICLE only
(envelope mode tracks no particle losses), so the match runs with
cost_solver="mp".

This lattice has a 6 mm aperture that clips ~4% of a moderate beam at the
seed.  The match (with MIN_TRANSMISSION 99.0 in the lattice) focuses the
beam through the aperture while holding transmission >= 99%.

Run headless:

    python examples/min_transmission/run_min_transmission.py

Equivalent CLI:

    python -m linac_gen.matching examples/min_transmission/min_transmission_demo.dat \
        --algorithm cmaes --max-iter 5 --cmaes-popsize 4 \
        --cost-solver mp --mp-n-particles 400 --space-charge \
        --energy 2.5 --current 5 --frequency 162.5 --report
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
from linac_gen.matching import match
from linac_gen.matching.engine import _run_mp

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = os.path.join(HERE, "min_transmission_demo.dat")

BEAM = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=400, distribution="waterbag",
    emit_nx=0.30, alpha_x=0.0, beta_x=0.6,
    emit_ny=0.30, alpha_y=0.0, beta_y=0.6,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)


def _transmission(lattice):
    res = _run_mp(lattice, copy.deepcopy(BEAM), space_charge=True,
                  n_particles=400, seed=42)
    t = getattr(res, "transmission", None)
    return float(t[-1]) if t is not None and len(t) else float("nan")


def main():
    print(f"Lattice: {LATTICE}\n")

    lat0, _ = parse_tracewin(LATTICE)
    print(f"Seed transmission (unmatched x0)  = {_transmission(lat0):.2f} %\n")

    lat, _ = parse_tracewin(LATTICE)
    r = match(lat, copy.deepcopy(BEAM), algorithm="cmaes", max_iter=5,
              cmaes_popsize=4, cost_solver="mp", space_charge=True,
              mp_n_particles=400, refine=False)

    print(f"Matched: cost {r.cost:.3e}  ({r.n_iter} evals, {r.elapsed_s:.1f}s)")
    print(f"Matched transmission              = {_transmission(lat):.2f} %\n")

    print("Constraint residuals (rms):")
    for c in r.constraints:
        import numpy as np
        res = r.per_constraint_residuals.get(c.label, np.array([0.0]))
        rms = float(np.sqrt(np.mean(res * res))) if res.size else 0.0
        print(f"  {c.label:<22s} {rms:.4e}")

    print("\nThe MIN_TRANSMISSION residual is ~0 once transmission >= 99%; it")
    print("would be large for any low-emittance solution that scraped the beam.")
    print("Try deleting the MIN_TRANSMISSION card from the .dat and re-running:")
    print("the matcher is then free to trade transmission for a lower epsilon.")


if __name__ == "__main__":
    main()
