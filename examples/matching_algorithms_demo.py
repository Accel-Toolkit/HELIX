"""Month-3 demo: selectable matcher optimisation algorithms.

Run from the repository root::

    python examples/matching_algorithms_demo.py

What this demo does
-------------------
Loads examples/matching_demo.dat — a short FODO cell with two linked
ADJUST cards on quad gradients and one SET_SIZE constraint (sigma_x = 4 mm
at the exit) — and runs HELIX's lattice-driven matcher three times, once
with each available optimisation algorithm:

  * least_squares          - local Levenberg-Marquardt (the default)
  * differential_evolution - global, gradient-free
  * dual_annealing         - global simulated annealing

It prints a side-by-side comparison of the matched quad gradient, the
final residual cost, the evaluation count and the wall time.

least_squares is local: fast, and only as good as its starting point.
The two global algorithms explore the whole bounded search box, so they
need finite min/max on every ADJUST card -- matching_demo.dat has them
("ADJUST QUAD 2 1 -30 30 ..."). A sigma-target is multi-modal in the
quad gradient, so a global search may legitimately settle in a different
(equally valid, ~zero-residual) basin than the local solver.

In the GUI
----------
Open examples/matching_demo.dat, go to the Matching tab -> AUTO-ADJUST
panel, pick the optimiser from the "Algorithm" dropdown, and click
Match. The dropdown defaults to least_squares (unchanged behaviour).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

from linac_gen.core.config import BeamConfig
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import MATCH_ALGORITHMS, match

REPO = Path(__file__).resolve().parent.parent
LATTICE = REPO / "examples" / "matching_demo.dat"


def _beam_cfg() -> BeamConfig:
    """The same 3-MeV proton beam used by matching_demo.py."""
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.3, alpha_z=0.0, beta_z=10.0,
    )


def main() -> None:
    print("=" * 80)
    print("HELIX Month-3 demo - selectable matcher algorithms")
    print("=" * 80)
    if not LATTICE.exists():
        raise SystemExit(f"missing lattice: {LATTICE}")

    lat0, meta = parse_tracewin(str(LATTICE))
    if meta.get("warnings"):
        print(f"  parse warnings: {meta['warnings']}")
    print(f"  lattice : {LATTICE.name} - {len(lat0.elements)} elements")
    print("  problem : SET_SIZE constraint, ADJUST on linked quad gradients")
    print()
    print(f"  {'algorithm':<24}{'success':>9}{'matched grad':>15}"
          f"{'cost':>13}{'evals':>8}{'time':>9}")
    print(f"  {'-' * 78}")

    for algo in MATCH_ALGORITHMS:
        # Fresh lattice + beam per algorithm so the runs are independent.
        lat = copy.deepcopy(lat0)
        cfg = _beam_cfg()
        t0 = time.time()
        res = match(lat, cfg, algorithm=algo, max_iter=200)
        dt = time.time() - t0
        grad = res.x_final[0] if len(res.x_final) else float("nan")
        print(f"  {algo:<24}{('OK' if res.success else 'FAIL'):>9}"
              f"{grad:>15.5f}{res.cost:>13.3e}{res.n_iter:>8d}{dt:>8.2f}s")

    print()
    print("  All three drive the SET_SIZE residual to ~zero. least_squares")
    print("  is the fastest (local); the global algorithms cost more")
    print("  evaluations but do not depend on the starting gradient.")
    print()
    print("  In the GUI: open matching_demo.dat -> Matching tab -> AUTO-ADJUST")
    print("  -> 'Algorithm' dropdown -> Match.")


if __name__ == "__main__":
    main()
