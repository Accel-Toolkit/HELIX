#!/usr/bin/env python
"""Demo: the sequential_scan matcher's two robustness controls.

1. **Reversal-threshold reference** -- whether the bracket-scan reverses
   direction by comparing the trial exit emittance to the *input* beam
   emittance (tight) or to the *seed-exit* emittance (the unmatched
   lattice's own exit emittance; natural when the lattice has intrinsic
   growth).  Run A vs Run B below.

2. **Hard loss-rejection** (`seqscan_reject_loss`) -- only meaningful with
   `cost_solver="mp"`: any trial step that drops transmission below the
   threshold is rolled back (best_x is NOT moved to it even if the cost
   fell) so the matcher cannot "win" on emittance by clipping halo at the
   tight aperture in this lattice.  Run C.

Run headless:

    python examples/sequential_scan/run_sequential_scan.py

Equivalent CLI (Run A / Run C):

    python -m linac_gen.matching examples/sequential_scan/seqscan_demo.dat \
        --algorithm sequential_scan --seqscan-threshold input \
        --energy 2.5 --current 5 --frequency 162.5 --report

    python -m linac_gen.matching examples/sequential_scan/seqscan_demo.dat \
        --algorithm sequential_scan --cost-solver mp --mp-n-particles 300 \
        --space-charge --seqscan-reject-loss --seqscan-loss-threshold-pct 99.0 \
        --energy 2.5 --current 5 --frequency 162.5 --report
"""
from __future__ import annotations

import copy
import os
import sys

# Ensure THIS repo's linac_gen wins over any other copy on the path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match
from linac_gen.matching.engine import _run_mp

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = os.path.join(HERE, "seqscan_demo.dat")

# Part 1 (reversal reference) uses a deliberately MISMATCHED beam so the
# lattice grows emittance and the reversal logic has work to do.  The
# 6 mm aperture is inert in envelope mode (no particle losses tracked).
BEAM_MISMATCHED = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=1000, distribution="waterbag",
    emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
    emit_ny=0.30, alpha_y=+2.0, beta_y=0.05,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)

# Part 2 (loss rejection) needs a beam that mostly fits the 6 mm aperture
# at the seed (~96% transmission) so the rejection rule can keep it there.
BEAM_MODERATE = BeamConfig(
    species="proton", energy=2.5, frequency=162.5, current=5.0,
    n_particles=400, distribution="waterbag",
    emit_nx=0.30, alpha_x=0.0, beta_x=0.6,
    emit_ny=0.30, alpha_y=0.0, beta_y=0.6,
    emit_z=0.40, alpha_z=0.0, beta_z=10.0,
)


def _scan(threshold):
    lat, _ = parse_tracewin(LATTICE)        # fresh copy -- match mutates
    r = match(lat, copy.deepcopy(BEAM_MISMATCHED), algorithm="sequential_scan",
              seqscan_passes=2, seqscan_steps=7, seqscan_step_frac=0.10,
              seqscan_reversal="both_grew", seqscan_threshold=threshold)
    return r


def _transmission(lattice):
    """Forward MP pass -> final transmission [%]."""
    res = _run_mp(lattice, copy.deepcopy(BEAM_MODERATE), space_charge=True,
                  n_particles=400, seed=42)
    t = getattr(res, "transmission", None)
    return float(t[-1]) if t is not None and len(t) else float("nan")


def _scan_mp(reject):
    lat, _ = parse_tracewin(LATTICE)
    r = match(lat, copy.deepcopy(BEAM_MODERATE), algorithm="sequential_scan",
              cost_solver="mp", space_charge=True, mp_n_particles=400,
              seqscan_passes=1, seqscan_steps=5, seqscan_step_frac=0.12,
              seqscan_reject_loss=reject, seqscan_loss_threshold_pct=95.0)
    return r, _transmission(lat)


def main():
    print(f"Lattice: {LATTICE}\n")

    print("== 1. Reversal-threshold reference (envelope) ==")
    print(f"{'threshold':<12}{'evals':>7}{'baseline':>14}{'final':>14}")
    print("-" * 47)
    for thr in ("input", "seed_exit"):
        r = _scan(thr)
        base = "n/a" if r.baseline_cost is None else f"{r.baseline_cost:.3e}"
        print(f"{thr:<12}{r.n_iter:>7}{base:>14}{r.cost:>14.3e}")
    print("\n  'input'     reverses when the trial exit eps exceeds the INPUT")
    print("              beam eps  (tight; good for coupling-resonance work).")
    print("  'seed_exit' reverses only when worse than the UNMATCHED lattice's")
    print("              own exit eps (natural for intrinsic-growth lattices).\n")

    print("== 2. Hard loss-rejection (multi-particle, 6 mm aperture) ==")
    lat0, _ = parse_tracewin(LATTICE)
    print(f"  seed transmission (unmatched x0)        = {_transmission(lat0):.2f} %")
    r_off, t_off = _scan_mp(reject=False)
    r_on, t_on = _scan_mp(reject=True)
    print(f"  reject_loss=OFF -> transmission {t_off:6.2f} %  (cost {r_off.cost:.3e})")
    print(f"  reject_loss=ON  -> transmission {t_on:6.2f} %  (cost {r_on.cost:.3e})")
    same = abs(t_off - t_on) < 0.5
    print()
    print("  reject_loss is a SAFETY RAIL: every trial step whose transmission")
    print("  falls below the 95% floor is rolled back (best_x is NOT moved to")
    print("  it even if cost dropped), so the matcher can never lower emittance")
    print("  by scraping beam at the 6 mm aperture.")
    if same:
        print("  On this gentle lattice the scan never needed a clipping step, so")
        print("  ON and OFF land in the same place -- the rail simply stayed")
        print("  inactive.  It bites on lattices where lowering eps DOES require")
        print("  clipping; see examples/min_transmission/ for that scenario, and")
        print("  try a tighter aperture / larger --seqscan-step-frac to trigger it.")
    else:
        print("  Here the rail changed the outcome: OFF accepted a lossier step")
        print("  that ON refused.")


if __name__ == "__main__":
    main()
