#!/usr/bin/env python3
"""Compare HELIX's numpy matrix tracker (v1) against the differentiable
torch tracker (v2).

The **v1 path** is the original numpy ``compute_transfer_matrix`` /
``compute_twiss`` — byte-unchanged in this HELIX_v2 tree, so it is
exactly what the original HELIX produces.  The **v2 path** is the new
``compute_transfer_matrix_torch`` / ``compute_twiss_torch``.  For every
pure-linear lattice the two must agree to ``--atol`` (default 1e-10).

Run::

    python scripts/compare_torch_matrix_tracking.py
    python scripts/compare_torch_matrix_tracking.py --lattice L2 --verbose

Exit code 0 if all pure-linear lattices agree, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.tracking.torch_tracking import (
    compute_transfer_matrix_torch, compute_twiss_torch,
)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _lattice(elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def lattices():
    """The L1-L4 comparison lattices (all pure-linear)."""
    return {
        "L1 single drift": _lattice([
            Drift("D", length=100.0),
        ]),
        "L2 FODO cell": _lattice([
            Drift("D1", length=200.0),
            Quadrupole("QF", length=100.0, gradient=8.0),
            Drift("D2", length=400.0),
            Quadrupole("QD", length=100.0, gradient=-8.0),
            Drift("D3", length=200.0),
        ]),
        "L3 solenoid channel": _lattice([
            Drift("D1", length=120.0),
            Solenoid("S1", length=80.0, field=0.5),
            Drift("D2", length=120.0),
            Solenoid("S2", length=80.0, field=-0.5),
            Drift("D3", length=120.0),
        ]),
        "L4 dipole cell": _lattice([
            Drift("D1", length=150.0),
            Edge("E1", pole_rotation=12.0, rho=1000.0),
            Dipole("B", angle=20.0, rho=1000.0, field_index=0.4),
            Edge("E2", pole_rotation=12.0, rho=1000.0),
            Drift("D2", length=150.0),
        ]),
    }


def compare(name, lat, ref, atol, verbose):
    """Run one lattice through both paths; return True if they agree."""
    M_v1 = compute_transfer_matrix(lat, ref)
    M_v2_t = compute_transfer_matrix_torch(lat, ref)
    M_v2 = M_v2_t.detach().numpy()
    dM = float(np.max(np.abs(M_v1 - M_v2)))

    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(200, 6))
    dX = float(np.max(np.abs((M_v1 @ X.T).T - (M_v2 @ X.T).T)))

    dtw = float("nan")
    tw_note = ""
    try:
        dtw = 0.0
        for pl in ("x", "y"):
            t1 = compute_twiss(M_v1, pl)
            t2 = compute_twiss_torch(M_v2_t, pl)
            dtw = max(dtw, max(abs(t1[k] - float(t2[k])) for k in t1))
    except ValueError as exc:
        tw_note = "  (Twiss N/A: " + str(exc).split("(")[0].strip() + ")"
        dtw = float("nan")

    ok = dM <= atol and dX <= 1e-8
    status = "OK" if ok else "FAIL"
    print(f"  {name:22s}  max|dM|={dM:.2e}  max|dX|={dX:.2e}  "
          f"max|dTwiss|={dtw:.2e}  [{status}]{tw_note}")
    if verbose:
        np.set_printoptions(precision=6, suppress=True)
        print(f"      v1 M =\n{M_v1}")
        print(f"      v2 M =\n{M_v2}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(
        description="v1 (numpy) vs v2 (torch) matrix-tracking comparison")
    p.add_argument("--atol", type=float, default=1e-10,
                   help="max allowed |v1-v2| matrix difference (default 1e-10)")
    p.add_argument("--lattice", default="all",
                   help="lattice key (L1/L2/L3/L4) or 'all'")
    p.add_argument("--verbose", action="store_true",
                   help="print the full v1 / v2 matrices")
    args = p.parse_args(argv)

    lats = lattices()
    if args.lattice != "all":
        lats = {k: v for k, v in lats.items()
                if k.split()[0] == args.lattice}
        if not lats:
            print(f"error: unknown lattice '{args.lattice}' "
                  f"(use L1/L2/L3/L4 or 'all')", file=sys.stderr)
            return 2

    ref = _ref()
    print("=" * 78)
    print("HELIX matrix tracking: v1 (numpy) vs v2 (torch, differentiable)")
    print(f"tolerance: atol = {args.atol:.0e}")
    print("=" * 78)
    all_ok = True
    for name, lat in lats.items():
        all_ok &= compare(name, lat, ref, args.atol, args.verbose)
    print("=" * 78)
    print("RESULT:", "all lattices agree — v2 reproduces v1 to tolerance"
          if all_ok else "MISMATCH — see the FAIL row(s) above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
