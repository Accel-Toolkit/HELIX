#!/usr/bin/env python
"""Demo: element failure analysis — impact ranking + fault recovery.

Sweeps single-element OFF failures over the demo lattice, ranks them by beam
impact (criticality), then re-tunes neighbouring cavities to try to recover the
beam after the worst failure (MYRRHA/LightWin-style local compensation).

Run headless:

    python examples/failure_analysis/run_failure_analysis.py

Equivalent CLI:

    python -m linac_gen failures examples/failure_analysis/demo.dat \
        --mode off --combination single --forward envelope --workers 1 \
        --energy 2.5 --freq 162.5 \
        --compensate --strategy k_out_of_n --k 2 \
        --comp-cost-solver envelope --comp-algorithm least_squares \
        --out failures.csv
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from linac_gen.cli.common import apply_beam_overrides
from linac_gen.core.config import BeamConfig
from linac_gen.failures import (CompensationConfig, FailureKind, FailureStudy,
                                compensate, enumerate_scenarios)
from linac_gen.io.tracewin_parser import parse_tracewin

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.join(HERE, "demo.dat")
BEAM = {"energy": 2.5, "frequency": 162.5, "current": 0.0,
        "emit_nx": 0.25, "emit_ny": 0.25, "emit_z": 0.30,
        "beta_x": 1.0, "beta_y": 1.0, "beta_z": 10.0}


def main():
    lat, _ = parse_tracewin(LAT)
    scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                          combination="single")
    print(f"Lattice: {LAT}")
    print(f"{len(names)} failable elements: {names}\n")

    study = FailureStudy(LAT, beam_overrides=BEAM, mode="envelope")
    res = study.run(scn, names, n2c, combination="single", serial=True)
    e0 = res.baseline["ref_w_kin"]
    print(f"Baseline exit energy = {e0:.4f} MeV\n")

    print("Criticality ranking (single-element OFF):")
    print(f"  {'element':<12}{'criticality':>12}{'ΔE [MeV]':>12}")
    print("  " + "-" * 36)
    for im in res.top(len(res.impacts)):
        print(f"  {im.scenario.element_names[0]:<12}{im.criticality:>12.4f}"
              f"{(im.d_energy_mev or 0.0):>12.4f}")

    worst = res.impacts[res.ranking[0]]
    print(f"\nWorst single failure: {worst.label}")
    print("Attempting local compensation (k_out_of_n, k=2)…")
    beam = BeamConfig()
    apply_beam_overrides(beam, BEAM)
    cfg = CompensationConfig(strategy="k_out_of_n", k=2,
                             algorithm="least_squares", cost_solver="envelope")
    cr = compensate(lat, beam, worst.scenario, n2c, res.baseline, cfg)
    e_after = cr.metrics_after.get("ref_w_kin")
    print(f"  compensators : {cr.compensator_names}")
    print(f"  recovered    : {cr.recovered}")
    print(f"  exit energy  : {e_after:.4f} MeV  (baseline {e0:.4f})")
    print(f"  settings     : "
          + ", ".join(f"{k}={v:.4g}" for k, v in cr.settings.items()))

    print("\nTakeaway: the failure sweep ranks which elements hurt the beam "
          "most;\ncompensation re-tunes neighbours to recover the design "
          "energy where\nthe surviving elements have the headroom to do so.")


if __name__ == "__main__":
    main()
